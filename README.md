
# Customer Churn Analysis

This project is a web application for analyzing customer churn data using a Flask web framework. It includes visualizations and metrics to help understand the factors influencing customer churn.

## Features

- ARPU Distribution by Churn Status
- Monthly Usage by Contract Type
- Confusion Matrix
- Classification Report
- Dataset Display
- Navigation between analysis and dataset pages

## Technologies Used

- Python
- Flask
- Pandas
- SQLAlchemy
- Matplotlib
- Seaborn
- Scikit-learn
- Bootstrap
- PostgreSQL

## Setup

### Prerequisites

- Python 3.x
- PostgreSQL

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/customer-churn-analysis.git
   cd customer-churn-analysis
2. Create and activate a virtual environment:   python3 -m venv venv
3. source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
4. Install the required packages:   python3 -m pip install Flask pandas sqlalchemy matplotlib seaborn scikit-learn psycopg2-binary
5. Set up the PostgreSQL database:
    * Install PostgreSQL using your preferred method (e.g., Homebrew on macOS).
    * Start PostgreSQL service:  brew services start postgresql@14
    * Access PostgreSQL prompt:  psql postgres
    * Create a new user and database:sql  CREATE USER myuser WITH PASSWORD 'mypassword';
    * CREATE DATABASE mydatabase OWNER myuser;
    * Exit PostgreSQL prompt:sql  \q
6. Create the CustomerBehavior table and insert sample data: sql  psql -d mydatabase -U myuser
```
CREATE TABLE CustomerBehavior (
    CustomerID INT PRIMARY KEY,
    Name VARCHAR(100),
    ARPU DECIMAL(5, 2),
    MRC DECIMAL(5, 2),
    ChurnStatus VARCHAR(3),
    MonthlyUsageGB DECIMAL(4, 1),
    ContractType VARCHAR(10),
    DeviceType VARCHAR(20),
    LastInteractionDate DATE,
    ProductHolding VARCHAR(255),
    ChurnPredictionScore DECIMAL(3, 2)
);

INSERT INTO CustomerBehavior (CustomerID, Name, ARPU, MRC, ChurnStatus, MonthlyUsageGB, ContractType, DeviceType, LastInteractionDate, ProductHolding, ChurnPredictionScore) VALUES
(1001, 'Robert French', 42.47, 26.89, 'No', 7.3, 'Annual', 'Smartphone', '2024-04-12', 'Home Security', 0.30),
(1002, 'Bryan Quinn', 77.04, 63.18, 'Yes', 28.9, 'Monthly', 'Tablet', '2024-01-01', 'TV, Home Security, Broadband', 0.75),
```
Running the Application
1. Navigate to the project directory:  cd customer-churn-analysis  
2. Run the Flask app:  python3 app.py   
3. Open your browser and go to http://127.0.0.1:5000/ to view the Customer Churn Analysis webpage. Use the navbar to navigate between the analysis page and the dataset page.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

This README file provides an overview of the project, lists the features, technologies used, setup instructions, and information on how to run the application. It also includes the project structure and license information.
