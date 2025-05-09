# Forecast Pro

A powerful demand forecasting application built with Python, offering both Flask and Streamlit interfaces.

## Features

- 📊 Interactive Dashboard
- 📈 Advanced Analysis Tools
- ⚙️ Customizable Settings
- 📱 Responsive Design
- 🔄 Multiple Forecasting Models
- 📁 Data Import/Export Support

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forecast-pro.git
cd forecast-pro
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Flask Version
```bash
python app.py
```
Access the application at: http://localhost:5000

### Streamlit Version
```bash
streamlit run app_streamlit.py
```
Access the application at: http://localhost:8501

## Project Structure

```
forecast-pro/
├── app.py                 # Flask application
├── app_streamlit.py       # Streamlit application
├── forecast.py            # Forecasting models
├── requirements.txt       # Project dependencies
├── static/               # Static files (CSS, JS)
├── templates/            # HTML templates
└── uploads/             # Uploaded data storage
```

## Data Format

The application accepts data in the following formats:
- CSV files
- Excel files (.xlsx)

Required columns:
- `date`: Date column
- `value`: Numeric values to forecast

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Flask and Streamlit
- Uses Plotly for visualizations
- Pandas for data manipulation 