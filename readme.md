# Workout Analyzer Backend

This is the backend service for the workout_analyzer application. It provides the necessary APIs and services to support the workout_analyzer platform.

## Features

- RESTful API endpoints using FastAPI
- Database integration
- Authentication and authorization
- Data processing and management
- Workout analysis and tracking

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/workout_analyzer-backend.git
cd workout_analyzer-backend
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Development mode:
```bash
python app.py
```

## API Documentation

The API documentation is available at:
- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Project Structure

```
workout_analyzer-backend/
├── app/
│   ├── api/           # API endpoints
│   ├── core/          # Core functionality
│   ├── models/        # Database models
│   ├── schemas/       # Pydantic schemas
│   ├── services/      # Business logic
│   └── utils/         # Utility functions
├── tests/             # Test files
├── .env              # Environment variables
└── requirements.txt   # Python dependencies
```