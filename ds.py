import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import datetime

# --- Configuration ---
DATA_PATH = "C:/climate-risk-dashboard/data/worldbank_climate_data_with_features.csv"
DEFAULT_LOCATION_COL = 'country'
DEFAULT_DATE_COL = 'date'


# --- Data Loading and Preprocessing ---

@st.cache_data  # Cache the data loading and initial processing
def load_data(file_path):
    """Loads, cleans, and preprocesses the World Bank data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {file_path}")
        st.info("Please run the `fetch_worldbank_data.py` script first to generate the data file.")
        return None

    # Basic Cleaning from fetch script (redundant but safe)
    if 'Unnamed: 0' in df.columns:  # Handle potential unnamed index column
        df = df.rename(columns={'Unnamed: 0': 'country'})  # Assuming it's country if multi-index wasn't saved properly
        if 'Unnamed: 1' in df.columns:
            df = df.rename(columns={'Unnamed: 1': 'date_str'})  # Assuming second level is date string
            df[DEFAULT_DATE_COL] = pd.to_datetime(df['date_str'])
            df = df.drop(columns=['date_str'])
        else:  # If only one unnamed column, need to parse country/date if they became index
            st.warning("CSV structure might be unexpected. Attempting to parse index.")
            # This part is tricky without seeing the exact CSV output if index wasn't reset
            # For now, assume columns exist as expected based on fetch script saving logic
            pass

    # Ensure date column is datetime
    if DEFAULT_DATE_COL in df.columns:
        df[DEFAULT_DATE_COL] = pd.to_datetime(df[DEFAULT_DATE_COL])
    else:
        st.error(f"Date column '{DEFAULT_DATE_COL}' not found. Cannot proceed with time series analysis.")
        return None

    # Ensure location column exists
    if DEFAULT_LOCATION_COL not in df.columns:
        st.error(f"Location column '{DEFAULT_LOCATION_COL}' not found.")
        return None

    # --- Feature Engineering (from fetch script - apply again or ensure consistency) ---
    # Ensure required columns for engineering exist
    required_eng_cols = ['population', 'land_area', 'urban_population_slums', 'forest_area_percent', 'gdp_per_capita']
    missing_eng_cols = [col for col in required_eng_cols if col not in df.columns]
    if missing_eng_cols:
        st.warning(f"Missing columns required for feature engineering: {missing_eng_cols}. Skipping.")
    else:
        epsilon = 1e-9  # Avoid division by zero
        df['population_density'] = df['population'] / (df['land_area'] + epsilon)
        df['urbanization_rate'] = df['urban_population_slums'] / (df['population'] + epsilon)
        df['forest_to_population_ratio'] = df['forest_area_percent'] / (df['population'] + epsilon)
        df['gdp_per_capita_normalized'] = df['gdp_per_capita'] / (df['population'] + epsilon)
        # Handle potential NaNs/Infs created by feature engineering
        eng_features = ['population_density', 'urbanization_rate', 'forest_to_population_ratio',
                        'gdp_per_capita_normalized']
        df[eng_features] = df[eng_features].replace([np.inf, -np.inf], np.nan)
        # Fill NaNs resulting from engineering (e.g., division by zero if population was 0)
        for col in eng_features:
            df[col] = df[col].fillna(df[col].mean())  # Or use 0 or median

    # --- Further Preprocessing for Modeling/Viz ---
    # Select numeric columns dynamically (excluding potentially non-numeric IDs if any)
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    # Exclude columns that shouldn't be transformed/scaled if necessary (e.g., year if it's numeric)
    # Example: if 'year' column exists: numeric_columns = [col for col in numeric_columns if col != 'year']

    # Fill missing values in numeric columns (e.g., from original data or failed engineering)
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())

    # Replace infinity values (just in case)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Refill NaNs potentially created by replacing infinities
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())

    # Apply log transformation (log1p handles 0, use for skewed data)
    # Important: Log transform can create -inf if input is 0, log1p(0) is 0.
    # Check for negative values before applying log1p
    df_processed = df.copy()  # Work on a copy for processing steps
    cols_to_log = []  # Keep track of columns we actually log-transformed
    for col in numeric_columns:
        if (df_processed[col] < 0).any():
            st.warning(
                f"Column '{col}' contains negative values. Log1p transform might produce NaNs or unexpected results. Skipping log transform for this column.")
        else:
            # Only apply log1p where it makes sense (e.g., counts, amounts)
            # Avoid applying to percentages or already normalized ratios unless distribution is highly skewed
            if col in ['population', 'gdp_per_capita', 'land_area']:  # Example selection
                # --- MODIFICATION START ---
                new_col_name = f"{col}_log1p"
                df_processed[new_col_name] = np.log1p(df_processed[col])  # Create NEW column
                cols_to_log.append(new_col_name)

    # Re-select numeric columns after potential renaming and filtering
    numeric_columns_processed = df_processed.select_dtypes(include=np.number).columns.tolist()

    # Scale numeric data using MinMaxScaler
    scaler = MinMaxScaler()
    # Avoid scaling columns that shouldn't be scaled (e.g., year)
    cols_to_scale = [c for c in numeric_columns_processed if c != 'year']  # Example exclusion
    if cols_to_scale:
        # Important: scaler.fit_transform returns numpy array, need to put back with correct columns
        scaled_data = scaler.fit_transform(df_processed[cols_to_scale])
        df_processed[cols_to_scale] = scaled_data
    else:
        st.warning("No columns identified for scaling.")

    # Final check for NaNs after all transformations
    nan_check_cols = df_processed.select_dtypes(include=np.number).columns
    if df_processed[nan_check_cols].isnull().values.any():
        st.warning("NaN values detected after processing. Attempting final fill with mean.")
        for col in nan_check_cols:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())

    return df_processed  # Return the processed dataframe


# --- Model Training ---
def train_model(df):
    """Trains a RandomForestRegressor model."""
    # Define features and target
    # Use engineered features and potentially log-transformed/scaled base features
    # Be careful about using log-transformed target if you want to interpret MSE easily
    features = ['population_density', 'urbanization_rate', 'forest_to_population_ratio', 'gdp_per_capita_normalized']
    target = 'gdp_per_capita'  # Using the original GDP per capita as target

    # Ensure target is in the original scale if features are transformed/scaled
    # Or predict the transformed target and inverse transform later

    # Check if features and target exist in the processed dataframe
    all_req_cols = features + [target]
    missing_cols = [col for col in all_req_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns required for model training: {missing_cols}. Check `load_data` processing steps.")
        return None, None, None, None, None, None

    # Check for NaN/Inf in features/target before training
    if df[features].isnull().values.any() or df[[target]].isnull().values.any():
        st.error("NaN values detected in features or target before training despite cleaning. Cannot proceed.")
        return None, None, None, None, None, None
    if np.isinf(df[features]).values.any() or np.isinf(df[[target]]).values.any():
        st.error("Infinite values detected in features or target before training. Cannot proceed.")
        return None, None, None, None, None, None

    X = df[features]
    y = df[target]

    # Split into train and test sets (consider time-based split if appropriate)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError as e:
        st.error(f"Error during train/test split: {e}. Check data shape and content.")
        st.error(f"X shape: {X.shape}, y shape: {y.shape}")
        return None, None, None, None, None, None

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10,
                                  min_samples_leaf=5)  # Added some hyperparameters
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Error during model fitting: {e}")
        return None, None, None, None, None, None

    # Make predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    importances = model.feature_importances_

    # Return model and results
    return model, X_test, y_test, y_pred, mse, importances


# --- Visualization Functions ---

def plot_choropleth_map(df, color_column, location_column=DEFAULT_LOCATION_COL):
    """Plots a choropleth map using country names."""
    if color_column not in df.columns:
        st.warning(f"Selected indicator '{color_column}' not found in the processed data for map.")
        return
    if location_column not in df.columns:
        st.error(f"Location column '{location_column}' not found.")
        return

    # Use the latest available data for each country for the map
    df_latest = df.loc[df.groupby(location_column)[DEFAULT_DATE_COL].idxmax()]

    if df_latest.empty:
        st.warning("No data available for the map.")
        return

    try:
        fig = px.choropleth(
            df_latest,
            locations=location_column,
            locationmode='country names',
            color=color_column,
            hover_name=location_column,
            hover_data={color_column: ':.2f', DEFAULT_DATE_COL: '%Y'},  # Format hover data
            color_continuous_scale=px.colors.sequential.Viridis,
            title=f"{color_column.replace('_', ' ').title()} by Country (Latest Data)",
        )
        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth')
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred creating the choropleth map: {e}")
        st.info("Check if country names are standard and the color column has valid numeric data.")


def plot_feature_importance(importances, feature_names):
    """Plots feature importances from the model."""
    if importances is None or feature_names is None:
        st.warning("Feature importances not available.")
        return
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                 title="Feature Importances for GDP per Capita Prediction")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)


def plot_ranked_bar_chart(df, indicator_column, n=15, location_column=DEFAULT_LOCATION_COL):
    """Plots ranked bar charts for top/bottom N countries for an indicator."""
    if indicator_column not in df.columns:
        st.warning(f"Indicator '{indicator_column}' not found for ranking.")
        return
    if not pd.api.types.is_numeric_dtype(df[indicator_column]):
        st.warning(f"Indicator '{indicator_column}' is not numeric and cannot be ranked.")
        return

    # Use the latest available data for ranking
    df_latest = df.loc[df.groupby(location_column)[DEFAULT_DATE_COL].idxmax()]
    df_latest = df_latest.dropna(subset=[indicator_column])  # Drop countries with NaN for this indicator
    df_ranked = df_latest.sort_values(by=indicator_column, ascending=False)

    if df_ranked.empty:
        st.warning(f"No valid data to rank for indicator '{indicator_column}'.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Top {n} Countries by {indicator_column.replace('_', ' ').title()}")
        fig_top = px.bar(df_ranked.head(n), x=indicator_column, y=location_column, orientation='h',
                         text=indicator_column)
        fig_top.update_traces(texttemplate='%{text:.2s}', textposition='outside')  # Format text
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.subheader(f"Bottom {n} Countries by {indicator_column.replace('_', ' ').title()}")
        # Filter out potential zeros or negative values if they skew the bottom ranking undesirably
        df_bottom = df_ranked[df_ranked[indicator_column] > 0].tail(
            n) if indicator_column != 'gdp_per_capita' else df_ranked.tail(n)
        if df_bottom.empty:
            st.info("No countries with positive values found for bottom ranking (or only zeros).")
        else:
            fig_bottom = px.bar(df_bottom.iloc[::-1], x=indicator_column, y=location_column, orientation='h',
                                # Reverse order for plot
                                text=indicator_column)
            fig_bottom.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig_bottom.update_layout(yaxis={'categoryorder': 'total descending'}, height=400)  # Order matches visual
            st.plotly_chart(fig_bottom, use_container_width=True)


def plot_relationship_scatter(df, x_col, y_col, color_col=None, hover_name=DEFAULT_LOCATION_COL):
    """Plots a scatter plot to show relationships between indicators."""
    if x_col not in df.columns or y_col not in df.columns:
        st.warning(f"One or both selected indicators ('{x_col}', '{y_col}') not found.")
        return
    if color_col and color_col not in df.columns:
        st.warning(f"Color indicator '{color_col}' not found. Plotting without color.")
        color_col = None

    # Use latest data for scatter plot to avoid overplotting time series
    df_latest = df.loc[df.groupby(hover_name)[DEFAULT_DATE_COL].idxmax()]
    df_latest = df_latest.dropna(subset=[x_col, y_col])

    if df_latest.empty:
        st.warning(f"No overlapping data found for indicators '{x_col}' and '{y_col}'.")
        return

    title = f"{y_col.replace('_', ' ').title()} vs. {x_col.replace('_', ' ').title()} (Latest Data)"
    try:
        fig = px.scatter(df_latest, x=x_col, y=y_col,
                         hover_name=hover_name,
                         color=color_col,
                         color_continuous_scale=px.colors.sequential.Plasma,
                         title=title,
                         trendline="ols",  # Add ordinary least squares trendline
                         trendline_color_override="red")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate scatter plot: {e}")


def plot_indicator_growth(df, indicator_column, countries, location_column=DEFAULT_LOCATION_COL,
                          date_column=DEFAULT_DATE_COL):
    """Plots the value of an indicator over time for selected countries."""
    if indicator_column not in df.columns:
        st.warning(f"Indicator '{indicator_column}' not found for growth plot.")
        return
    if not countries:
        st.info("Select one or more countries to visualize growth.")
        return

    df_filtered = df[df[location_column].isin(countries)].sort_values(by=date_column)

    if df_filtered.empty:
        st.warning(f"No data found for the selected countries and indicator '{indicator_column}'.")
        return

    # Optional: Calculate and plot percentage change (growth rate) instead of absolute value
    # df_filtered['growth_rate'] = df_filtered.groupby(location_column)[indicator_column].pct_change() * 100
    # y_col_to_plot = 'growth_rate'
    # title_suffix = "Growth Rate (%)"

    y_col_to_plot = indicator_column  # Plot absolute value for now
    title_suffix = "Over Time"

    title = f"{indicator_column.replace('_', ' ').title()} for Selected Countries {title_suffix}"
    try:
        fig = px.line(df_filtered, x=date_column, y=y_col_to_plot, color=location_column,
                      title=title, markers=True)
        fig.update_layout(legend_title_text='Country')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate growth plot: {e}")


def plot_predictions(y_test, y_pred):
    """Plots true vs predicted values."""
    if y_test is None or y_pred is None:
        st.warning("Prediction data not available.")
        return
    df_pred = pd.DataFrame({'True Values': y_test, 'Predictions': y_pred})
    fig = px.scatter(df_pred, x="True Values", y="Predictions",
                     title="GDP per Capita: True vs. Predicted Values (Test Set)",
                     trendline="ols", trendline_color_override="red")
    fig.add_shape(type='line', x0=df_pred['True Values'].min(), y0=df_pred['True Values'].min(),
                  x1=df_pred['True Values'].max(), y1=df_pred['True Values'].max(),
                  line=dict(color='gray', dash='dash'))  # Add y=x line
    st.plotly_chart(fig, use_container_width=True)


# --- Streamlit App Main Function ---
def main():
    st.set_page_config(layout="wide", page_title="Climate Risk & Economic Insights")

    st.title("Climate Risk and Economic Insights Dashboard")
    st.markdown("""
        Explore World Bank data on various economic and climate-related indicators.
        Visualize trends, compare countries, and see model predictions.
        *Note: Data processing includes log transformation and scaling for some visualizations and modeling.*
    """)

    # Load Data
    df_processed = load_data(DATA_PATH)

    if df_processed is None:
        st.stop()  # Stop execution if data loading failed

    # Train Model (only if data loaded successfully)
    model, X_test, y_test, y_pred, mse, importances = train_model(df_processed)

    # --- Sidebar ---
    st.sidebar.header("Map Options")
    # Get numeric columns suitable for map coloring (exclude obviously unsuitable ones if needed)
    numeric_cols_map = sorted([col for col in df_processed.columns if
                               pd.api.types.is_numeric_dtype(df_processed[col]) and col not in [
                                   DEFAULT_DATE_COL]])  # Exclude date if numeric
    default_map_index = numeric_cols_map.index('gdp_per_capita') if 'gdp_per_capita' in numeric_cols_map else 0
    map_indicator = st.sidebar.selectbox(
        "Select Indicator for Map:",
        options=numeric_cols_map,
        index=default_map_index
    )

    # --- Main Area Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 World Map",
        "📊 Country Rankings",
        "🔗 Indicator Relationships",
        "📈 Growth Over Time",
        "🤖 Model Insights"
    ])

    with tab1:
        st.header("Global Indicator Overview")
        st.markdown(
            f"Visualizing **{map_indicator.replace('_', ' ').title()}** across the globe using the latest available data for each country.")
        plot_choropleth_map(df_processed, color_column=map_indicator, location_column=DEFAULT_LOCATION_COL)

    with tab2:
        st.header("Compare Countries")
        rank_indicator = st.selectbox(
            "Select Indicator for Ranking:",
            options=numeric_cols_map,  # Use the same list as map
            index=default_map_index  # Default to GDP
        )
        plot_ranked_bar_chart(df_processed, indicator_column=rank_indicator, n=15, location_column=DEFAULT_LOCATION_COL)

    with tab3:
        st.header("Explore Relationships Between Indicators")
        st.markdown("Select two indicators to see their relationship using the latest data point for each country.")
        col1, col2 = st.columns(2)
        with col1:
            x_axis_indicator = st.selectbox("Select X-axis Indicator:", options=numeric_cols_map,
                                            index=numeric_cols_map.index(
                                                'population_density') if 'population_density' in numeric_cols_map else 1)
        with col2:
            y_axis_indicator = st.selectbox("Select Y-axis Indicator:", options=numeric_cols_map,
                                            index=numeric_cols_map.index(
                                                'gdp_per_capita') if 'gdp_per_capita' in numeric_cols_map else 0)

        # Optional: Color by a third indicator
        color_indicator = st.selectbox("Optional: Color points by Indicator:", options=[None] + numeric_cols_map,
                                       index=0)

        plot_relationship_scatter(df_processed, x_col=x_axis_indicator, y_col=y_axis_indicator,
                                  color_col=color_indicator, hover_name=DEFAULT_LOCATION_COL)

    with tab4:
        st.header("Indicator Growth Trends")
        growth_indicator = st.selectbox(
            "Select Indicator to Track:",
            options=numeric_cols_map,
            index=default_map_index
        )
        all_countries = sorted(df_processed[DEFAULT_LOCATION_COL].unique())
        # Select default countries (e.g., major economies or diverse examples)
        default_countries = [c for c in ['United States', 'China', 'India', 'Germany', 'Brazil', 'Nigeria'] if
                             c in all_countries]
        selected_countries = st.multiselect(
            "Select Countries:",
            options=all_countries,
            default=default_countries
        )
        plot_indicator_growth(df_processed, indicator_column=growth_indicator, countries=selected_countries,
                              location_column=DEFAULT_LOCATION_COL, date_column=DEFAULT_DATE_COL)

    with tab5:
        st.header("GDP per Capita Prediction Insights")
        if model is not None and importances is not None:
            st.subheader("Model Feature Importances")
            st.markdown("Which features were most influential in predicting GDP per capita?")
            feature_names = ['population_density', 'urbanization_rate', 'forest_to_population_ratio',
                             'gdp_per_capita_normalized']  # Must match features used in train_model
            plot_feature_importance(importances, feature_names)

            st.subheader("Model Prediction Performance")
            st.markdown("How well did the model predict GDP per capita on unseen test data?")
            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.caption("Lower MSE indicates better model performance.")
            plot_predictions(y_test, y_pred)
        else:
            st.error("Model could not be trained. Cannot display insights.")

    # Optional: Add a data explorer tab
    st.divider()
    with st.expander("View Processed Data Sample"):
        st.dataframe(df_processed.head())
        st.dataframe(df_processed.describe())


if __name__ == "__main__":
    main()
