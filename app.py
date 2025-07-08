import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import joblib # Import joblib to load the model

st.set_page_config(layout="wide") # Optional: Use wide layout

st.title('Extrovert vs. Introvert Behavior Data Analysis')

# --- Data Loading and Processing ---
# In a deployed Streamlit app, you'd typically load your data here.
# Assuming you have saved your processed 'data' DataFrame to 'processed_data.csv'
try:
    data = pd.read_csv('processed_data.csv')
    # Convert 'Personality' to string for better color labels in Plotly Express
    data['Personality'] = data['Personality'].astype(str)
    # Ensure boolean columns are treated as int for Plotly graph_objs if needed
    for col in ['Stage_fear_Yes', 'Drained_after_socializing_Yes']:
        if col in data.columns:
             data[col] = data[col].astype(int)

except FileNotFoundError:
    st.error("Error: 'processed_data.csv' not found. Please ensure it's in the same directory as app.py.")
    st.stop() # Stop the app if data loading fails
except Exception as e:
    st.error(f"An error occurred during data loading or processing: {e}")
    st.stop()

# --- Load the trained model ---
model = None # Initialize model to None
try:
    model = joblib.load('logistic_regression_model.pkl')
    st.sidebar.success("Machine learning model loaded successfully!")
except FileNotFoundError:
    st.sidebar.warning("Warning: 'logistic_regression_model.pkl' not found. The prediction functionality will not be available.")
except Exception as e:
    st.sidebar.error(f"An error occurred while loading the model: {e}")


# --- Interactive Scatter Plot ---
st.header('Interactive Scatter Plot: Time Spent Alone vs. Post Frequency')

fig_scatter = px.scatter(data,
                 x='Time_spent_Alone',
                 y='Post_frequency',
                 size='Friends_circle_size',
                 color='Personality',
                 hover_name=data.index,
                 title='Tempo Sozinho vs. Frequência de Postagem (Colorido por Personalidade, Tamanho por Círculo de Amizade)'
                )
st.plotly_chart(fig_scatter, use_container_width=True)


# --- Interactive Box/Violin Plots for Numeric Variables with Selection ---
st.header('Distribution of Numeric Variables by Personality (Box/Violin Plots)')

numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
target_column = 'Personality' # Already converted to string above

# Add a selectbox for the user to choose the numeric variable
selected_numeric_col = st.selectbox(
    'Select a numeric variable to visualize:',
    numeric_columns
)

# Generate the Box/Violin plot for the selected column
if selected_numeric_col:
    st.subheader(f'Distribution of {selected_numeric_col} by Personality')

    fig_bv = go.Figure()

    # Add initial Box plot for the selected numeric column by Personality
    fig_bv.add_trace(go.Box(
        y=data[data[target_column] == '0'][selected_numeric_col], # Use string '0' and '1'
        name='Extrovert (0)'
    ))
    fig_bv.add_trace(go.Box(
        y=data[data[target_column] == '1'][selected_numeric_col], # Use string '0' and '1'
        name='Introvert (1)'
    ))

    # Add Violin plot traces, initially hidden
    fig_bv.add_trace(go.Violin(
        y=data[data[target_column] == '0'][selected_numeric_col],
        name='Extrovert (0)',
        visible=False # Initially hidden
    ))
    fig_bv.add_trace(go.Violin(
        y=data[data[target_column] == '1'][selected_numeric_col],
        name='Introvert (1)',
        visible=False # Initially hidden
    ))


    fig_bv.layout.update(
       updatemenus = [
          go.layout.Updatemenu(
             type = "buttons", direction = "left", buttons=list(
                [
                   dict(
                       args = [{"visible": [True, True, False, False], "type": "box"}], # Show Box, Hide Violin
                       label = "Box",
                       method = "restyle"
                   ),
                   dict(
                       args = [{"visible": [False, False, True, True], "type": "violin"}], # Hide Box, Show Violin
                       label = "Violin",
                       method = "restyle"
                   )
                ]
             ),
             pad = {"r": 2, "t": 2},
             showactive = True,
             x = 0.11,
             xanchor = "left",
             y = 1.1,
             yanchor = "top"
          ),
       ],
       title = f'Distribuição de {selected_numeric_col} por Personalidade', # Title is set in st.subheader
       yaxis_title = selected_numeric_col
    )
    st.plotly_chart(fig_bv, use_container_width=True)

# --- Sunburst Chart Section ---
st.header('Gráfico Sunburst das Variáveis Categóricas por Personalidade')

# Legenda das Abreviações
st.markdown("""
**Legenda das Abreviações:**

*   **TSA:** Time_spent_Alone (Tempo Sozinho)
*   **SEA:** Social_event_attendance (Participação em eventos sociais)
*   **GO:** Going_outside (Sair)
*   **FCS:** Friends_circle_size (Tamanho do círculo de amizade)
*   **PF:** Post_frequency (Frequência de Postagem)
""")


# Load the categorized data
try:
    categorized_data = pd.read_csv('categorized_data.csv')
    # Ensure 'Personality' is treated as string for plotly if needed
    if categorized_data['Personality'].dtype != 'object':
         categorized_data['Personality'] = categorized_data['Personality'].astype(str)

    # Identify the available categorical columns (excluding Personality)
    available_categorical_columns = [col for col in categorized_data.columns if col.endswith('_category')]

    # Add a multiselect widget for interactive column selection
    selected_path_columns = st.multiselect(
        'Selecione as variáveis para o caminho do gráfico Sunburst (a ordem importa):',
        available_categorical_columns,
        default=available_categorical_columns # Default selection includes all categorized columns
    )

    # Always include 'Personality' as the first level
    path_columns_with_personality = ['Personality'] + selected_path_columns

    # Ensure at least 'Personality' is selected to avoid errors
    if not selected_path_columns:
        st.warning("Por favor, selecione pelo menos uma variável categórica além de 'Personality'.")
    else:
        # Calculate counts for the sunburst chart based on selected columns
        sunburst_data = categorized_data.groupby(path_columns_with_personality).size().reset_index(name='count')

        # Create the sunburst chart
        fig_sunburst = px.sunburst(sunburst_data,
                                   path=path_columns_with_personality,
                                   values='count',
                                   title='Distribuição Categórica por Personalidade') # Title can be adjusted

        # Display the sunburst chart in Streamlit
        st.plotly_chart(fig_sunburst, use_container_width=True)

except FileNotFoundError:
    st.error("Error: 'categorized_data.csv' not found. Please ensure it's in the same directory as app.py.")
except Exception as e:
    st.error(f"An error occurred while creating the Sunburst chart: {e}")


# --- Machine Learning Prediction Section ---
st.header('Prever Personalidade')

if model: # Only show this section if the model was loaded successfully
    st.write("Insira os valores para cada característica para prever a personalidade (0: Extrovert, 1: Introvert).")

    # Create input fields for each feature used in the model
    # Get the feature names from the model's training data (X_train)
    # Assuming X_train was created in a previous cell and is available in the environment
    try:
        feature_names = X_train.columns.tolist()
    except NameError:
        st.error("Error: X_train not found. Please run the cell that trains the model first.")
        feature_names = [] # Set empty list if X_train is not defined


    input_data = {}
    if feature_names: # Only create inputs if feature names are available
        for feature in feature_names:
            # Try to get a default value from the 'data' DataFrame
            default_value = None
            if feature in data.columns:
                if data[feature].dtype in ['float64', 'int64']:
                     default_value = float(data[feature].mean())
                elif data[feature].dtype == 'bool':
                     default_value = bool(data[feature].mode()[0])
                else:
                     default_value = str(data[feature].mode()[0])

            if data[feature].dtype in ['float64', 'int64']:
                 input_data[feature] = st.number_input(f'Enter value for {feature}', value=default_value)
            elif data[feature].dtype == 'bool':
                 input_data[feature] = st.checkbox(f'Select if {feature} is True', value=default_value)
            else:
                 input_data[feature] = st.text_input(f'Enter value for {feature}', value=default_value)


        # Create a button to trigger the prediction
        if st.button('Prever Personalidade'):
            try:
                # Convert input data to a pandas DataFrame in the correct order
                input_df = pd.DataFrame([input_data])

                # Ensure the order of columns in input_df matches the order used during training
                input_df = input_df[feature_names]

                # Convert boolean columns back to correct type if necessary for prediction
                for col in ['Stage_fear_Yes', 'Drained_after_socializing_Yes']:
                    if col in input_df.columns:
                         input_df[col] = input_df[col].astype(bool) # Convert back to boolean

                # Make the prediction
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df) # Get probabilities

                # Display the prediction result
                predicted_class = "Introvert" if prediction[0] == 1 else "Extrovert"
                st.subheader(f"Predicted Personality: **{predicted_class}**")
                st.write(f"Prediction Probability (Extrovert): {prediction_proba[0][0]:.4f}")
                st.write(f"Prediction Probability (Introvert): {prediction_proba[0][1]:.4f}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Machine learning model not loaded. Prediction functionality is not available.")


# --- Optional: Add more sections for categorical variables, etc. ---
# st.header('Distribution of Categorical Variables')
# ... add code for count plots or other categorical visualizations ...
