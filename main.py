from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pytesseract
from PIL import Image
import io
import re
from datetime import datetime
import json
from typing import Optional

# Create FastAPI app
app = FastAPI(
    title="Immigration Document OCR API",
    description="Extract text and classify immigration documents",
    version="1.0.0"
)

# Add CORS middleware to allow website integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your website domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class OCRResult(BaseModel):
    success: bool
    document_type: str
    confidence: float
    extracted_text: str
    structured_data: dict
    error_message: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    message: str

# Simple Document Classifier (same logic as your Streamlit app)
class DocumentClassifier:
    def __init__(self):
        self.training_data = [
            ("passport number personal details republic india", "passport"),
            ("visa entry permit immigration canada", "visa"),
            ("work permit employment authorization", "permit"),
            ("birth certificate date of birth", "certificate"),
            ("driver license identification card", "identification"),
        ]
    
    def classify_document(self, text):
        text_lower = text.lower()
        
        # Simple keyword-based classification
        if any(word in text_lower for word in ["passport", "republic", "travel document"]):
            return {"document_type": "passport", "confidence": 0.9}
        elif any(word in text_lower for word in ["visa", "entry permit", "immigration"]):
            return {"document_type": "visa", "confidence": 0.9}
        elif any(word in text_lower for word in ["work permit", "employment"]):
            return {"document_type": "permit", "confidence": 0.8}
        elif any(word in text_lower for word in ["certificate", "birth", "marriage"]):
            return {"document_type": "certificate", "confidence": 0.8}
        elif any(word in text_lower for word in ["license", "identification"]):
            return {"document_type": "identification", "confidence": 0.8}
        else:
            return {"document_type": "unknown", "confidence": 0.5}

# Data Extractor (same logic as your Streamlit app)
class DataExtractor:
    def __init__(self):
        self.patterns = {
            "passport_number": r'\b[A-Z]{1,2}[0-9]{6,8}\b',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            "name": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            "visa_number": r'\b[A-Z0-9]{8,12}\b',
            "phone": r'\+?[0-9]{10,13}',
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        }
    
    def extract_structured_data(self, text, document_type):
        base = {
            "extraction_date": datetime.now().isoformat(),
            "raw_text_length": len(text),
            "document_type": document_type
        }
        
        # Find passport number
        if document_type == "passport":
            m = re.search(self.patterns['passport_number'], text, re.IGNORECASE)
            if m:
                base['passport_number'] = m.group()
        
        # Find visa number
        if document_type == "visa":
            m = re.search(self.patterns['visa_number'], text, re.IGNORECASE)
            if m:
                base['visa_number'] = m.group()
        
        # Find dates
        dates = re.findall(self.patterns['date'], text)
        if dates:
            base['dates_found'] = dates[:3]
        
        # Find names
        names = re.findall(self.patterns['name'], text)
        if names:
            base['names_found'] = names[:2]
        
        # Find email
        email = re.search(self.patterns['email'], text, re.IGNORECASE)
        if email:
            base['email'] = email.group()
        
        return base

# Initialize processors
classifier = DocumentClassifier()
extractor = DataExtractor()

# API Endpoints

@app.get("/", response_model=HealthCheck)
async def root():
    """Health check endpoint"""
    return HealthCheck(
        status="online",
        message="Immigration Document OCR API is running!"
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    try:
        # Test Tesseract
        pytesseract.get_tesseract_version()
        return HealthCheck(
            status="healthy",
            message="All systems operational"
        )
    except Exception as e:
        return HealthCheck(
            status="error",
            message=f"Tesseract not available: {str(e)}"
        )

@app.post("/process-document", response_model=OCRResult)
async def process_document(file: UploadFile = File(...)):
    """
    Process an uploaded immigration document
    
    - **file**: Upload a PNG or JPG image of an immigration document
    - Returns: Extracted text, document type, and structured data
    """
    
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(
                status_code=400, 
                detail="Only JPEG and PNG images are supported"
            )
        
        # Validate file size (max 10MB)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB"
            )
        
        # Open image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(image, lang="eng+hin")
        
        # Check if text was extracted
        if not extracted_text.strip():
            return OCRResult(
                success=False,
                document_type="unknown",
                confidence=0.0,
                extracted_text="",
                structured_data={},
                error_message="No text could be extracted from the image"
            )
        
        # Classify document type
        classification = classifier.classify_document(extracted_text)
        
        # Extract structured data
        structured_data = extractor.extract_structured_data(
            extracted_text, 
            classification['document_type']
        )
        
        return OCRResult(
            success=True,
            document_type=classification['document_type'],
            confidence=classification['confidence'],
            extracted_text=extracted_text.strip()[:2000],  # Limit text length
            structured_data=structured_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return OCRResult(
            success=False,
            document_type="unknown",
            confidence=0.0,
            extracted_text="",
            structured_data={},
            error_message=f"Processing error: {str(e)}"
        )

@app.post("/extract-text-only")
async def extract_text_only(file: UploadFile = File(...)):
    """
    Simple endpoint that only extracts text (no classification)
    """
    try:
        if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG supported")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        text = pytesseract.image_to_string(image, lang="eng+hin")
        
        return {
            "success": True,
            "extracted_text": text.strip(),
            "character_count": len(text.strip())
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
