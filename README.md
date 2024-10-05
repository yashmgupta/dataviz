# dataviz

# Professional Data Analysis and Visualization App

This is a **data analysis and visualization web application** built using **Streamlit**. The app allows users to upload CSV or Excel files and provides automatic data analysis, advanced data visualization options, and advanced research-level analysis features like **PCA**, **K-Means Clustering**, **Outlier Detection**, and **Regression Analysis**.

## Features

- **Upload CSV or Excel files** for analysis.
- **Automatic Data Analysis**:
  - Dataset overview: shape, columns, missing values, data types, and summary statistics.
- **Advanced Analysis Options**:
  - **Principal Component Analysis (PCA)**: Reduce dimensionality and visualize the principal components.
  - **K-Means Clustering**: Identify clusters within the dataset.
  - **Outlier Detection**: Use Z-score to identify outliers.
  - **Regression Analysis**: Perform linear regression analysis and display coefficients, intercept, and R^2 score.
- **Plotting Options**:
  - **Scatter Plot, Line Chart, Bar Chart, Histogram, Box Plot, Heatmap, Pair Plot, Violin Plot, Area Chart, Count Plot**.
- **Download the Processed Data** as a CSV file.

## Installation

To run this application locally, you need to have Python installed along with some required libraries. Follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yashmgupta/dataviz
   cd dataviz
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the application:
   ```sh
   streamlit run app.py
   ```

4. Open your browser to `http://localhost:8501` to view the app.

## Dependencies

The following Python packages are required to run the application:

- `streamlit`
- `pandas`
- `matplotlib`
- `seaborn`
- `openpyxl`
- `numpy`
- `scikit-learn`
- `scipy`

These dependencies can be installed using the `requirements.txt` file provided in the repository.

## How to Use

1. **Upload Your Dataset**: Use the file uploader to select a CSV or Excel file for analysis.
2. **Automatic Data Analysis**: View basic information about the dataset, including missing values, data types, and summary statistics.
3. **Advanced Analysis Options**: Use the sidebar to perform advanced analysis:
   - **PCA**: Select the number of components to reduce the dataset and visualize the results.
   - **K-Means Clustering**: Choose the number of clusters to segment your data.
   - **Outlier Detection**: Identify and display outliers in the dataset.
   - **Regression Analysis**: Select the target and feature variables to run a linear regression model.
4. **Plotting Options**: Select columns and visualize your data with different types of plots.
5. **Download the Processed Data**: After analysis, you can download the data as a CSV file.

## License

This project is licensed under the **MIT License**. See the LICENSE file for more details.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to fork the repository and submit a pull request.

## Acknowledgments

- **Streamlit** for making it easy to build web apps for data analysis.
- **Seaborn** and **Matplotlib** for the visualizations.
- **Scikit-Learn** for providing the tools for clustering, PCA, and regression analysis.

## Contact

If you have any questions, feel free to reach out at yashmunnalalg@nu.ac.th.

