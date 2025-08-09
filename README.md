# ML & Drone Systems Research Papers - Setup Guide

## 🚀 Quick Start

### 1. Installation

```bash
# Create a virtual environment (recommended)
python -m venv arxiv_env
source arxiv_env/bin/activate  # On Windows: arxiv_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Application

```bash
# Run the Streamlit app
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### 3. Using the Data Collector (Optional)

```bash
# Run the data collector script to generate a fresh dataset
python arxiv_collector.py
```

## 📁 Project Structure

```
├── arxiv_collector.py      # Data collection script
├── streamlit_app.py        # Main web application
├── requirements.txt        # Dependencies
├── README.md              # This file
└── datasets/              # Generated datasets (created automatically)
```

## 🔧 Features

### Data Collection
- **Automated arXiv Search**: Searches for papers related to machine learning and adaptive behavior in drone systems
- **Comprehensive Metadata**: Collects title, summary, authors, categories, publication dates, and PDF links
- **Rate-Limited API Calls**: Respectful to arXiv's API with built-in delays
- **Error Handling**: Robust error handling for network issues and API limits

### Web Interface
- **Interactive Dashboard**: Beautiful Streamlit interface with metrics and visualizations
- **Real-time Search**: Live arXiv search functionality
- **Advanced Filtering**: Filter by categories, search terms, and date ranges
- **PDF Downloads**: Direct links to download research papers
- **Dataset Export**: Download the entire dataset as CSV
- **Responsive Design**: Works on desktop and mobile devices

### Data Features
- **Rich Metadata**: Each paper includes:
  - Title and summary
  - Author information
  - Publication and update dates
  - arXiv categories and IDs
  - Direct PDF and abstract URLs
  - DOI information (when available)

## 🎯 Usage Examples

### Search Query Customization

The default search focuses on:
- Machine learning techniques (deep learning, reinforcement learning, neural networks)
- Drone/UAV systems (drones, UAVs, quadcopters, multirotors)
- Adaptive behaviors (control, autonomous navigation, adaptive systems)

You can modify the search query in the Streamlit interface or in the collector script.

### Sample Queries
- `"reinforcement learning" AND UAV AND navigation`
- `"deep learning" AND drone AND "object detection"`
- `"neural network" AND quadcopter AND control`

## 📊 Dataset Schema

| Column | Description |
|--------|-------------|
| title | Paper title |
| summary | Abstract/summary |
| authors | Comma-separated list of authors |
| published_date | Publication date (YYYY-MM-DD) |
| updated_date | Last update date |
| primary_category | Primary arXiv category |
| categories | All arXiv categories |
| arxiv_id | arXiv identifier |
| pdf_url | Direct PDF download link |
| entry_url | arXiv abstract page URL |
| doi | Digital Object Identifier (if available) |

## 🔍 Categories Explained

Common arXiv categories you'll encounter:
- **cs.RO**: Robotics
- **cs.LG**: Machine Learning
- **cs.AI**: Artificial Intelligence
- **cs.CV**: Computer Vision
- **cs.MA**: Multiagent Systems
- **cs.SY**: Systems and Control

## 🛠️ Customization

### Modifying Search Parameters
Edit the query in `arxiv_collector.py`:

```python
query = """
    YOUR_CUSTOM_SEARCH_TERMS_HERE
"""
```

### Changing Result Limits
Adjust the `max_results` parameter (arXiv API limit is typically 2000):

```python
papers_data = search_arxiv_papers(query, max_results=500)
```

### Adding New Features
The modular design allows easy extension:
- Add new metadata fields in the `search_arxiv_papers` function
- Create additional visualizations in the Streamlit app
- Implement different export formats (JSON, Excel, etc.)

## 🚨 Important Notes

### API Limitations
- arXiv API has rate limits (be respectful)
- Large queries (>200 papers) may take several minutes
- Network issues can interrupt long searches

### Data Quality
- Summaries are author-provided abstracts
- Categories are assigned by authors and arXiv moderators
- Publication dates reflect arXiv submission, not journal publication

### PDF Access
- All PDF links are direct downloads from arXiv
- PDFs are freely accessible (arXiv's mission)
- Large PDFs may take time to download

## 🎨 Customizing the Interface

### Themes and Styling
The app uses custom CSS that can be modified in the Streamlit app. Key areas:
- Color scheme (currently blue-based)
- Card layouts for papers
- Typography and spacing

### Adding Visualizations
Easy to add new charts using Plotly:

```python
fig = px.scatter(df, x='published_date', y='category', title='Publication Timeline')
st.plotly_chart(fig)
```

## 🤝 Contributing

Feel free to enhance the application:
1. Add new search filters
2. Implement additional visualizations
3. Add export formats
4. Improve error handling
5. Add caching mechanisms

## 📄 License

This project is for educational and research purposes. Please respect arXiv's terms of use and rate limits.

## 🆘 Troubleshooting

### Common Issues

1. **"No papers found"**
   - Check your internet connection
   - Try broader search terms
   - Reduce the number of results

2. **Slow loading**
   - Large result sets take time
   - arXiv API has rate limits
   - Consider using sample data for testing

3. **PDF download issues**
   - Links are direct to arXiv
   - Check your browser's download settings
   - Large PDFs may timeout

4. **Installation problems**
   - Ensure Python 3.8+ is installed
   - Use virtual environment
   - Check all dependencies are installed

### Support
For issues with the arXiv API, consult: https://arxiv.org/help/api/
For Streamlit issues, see: https://docs.streamlit.io/
