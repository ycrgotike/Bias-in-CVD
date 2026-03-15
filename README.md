# CVD Prediction Flask Application

## Project Description
This project is a Flask application designed for predicting Cardiovascular Disease (CVD) using gender-specific machine learning models. It aims to provide a user-friendly interface to input patient data and receive predictions regarding the likelihood of CVD.

## Features
- Gender-specific predictive models for better accuracy.
- Simple user interface for data input.
- Real-time predictions based on input data.
- Comprehensive output detailing risk factors and recommendations.

## Installation Instructions
1. Clone the repository:
   ```
   git clone https://github.com/ycrgotike/Bias-in-CVD.git
   cd Bias-in-CVD
   ```
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python app.py
   ```

## Project Structure
```
Bias-in-CVD/
├── app.py            # Main application file
├── models/           # Directory for storing machine learning models
├── templates/        # HTML templates for the web app
└── static/          # Static files (CSS, JavaScript, images)
```

## Usage Guidelines
1. Open your web browser and navigate to `http://localhost:5000`.
2. Fill in the required patient data fields.
3. Click on the 'Predict' button to receive your CVD risk assessment.
4. Review the output for insights and recommendations.

## Conclusion
This Flask application serves as a reliable tool for predicting the risk of CVD using advanced machine learning techniques tailored for gender-specific data.