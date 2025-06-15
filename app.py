import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

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


# --- Optional: Add more sections for categorical variables, etc. ---
# st.header('Distribution of Categorical Variables')
# ... add code for count plots or other categorical visualizations ...
