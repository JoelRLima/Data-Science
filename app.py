import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import joblib

st.set_page_config(layout="wide")  # Layout amplo

st.title('Extrovert vs. Introvert Behavior Data Analysis')

# --- Carregamento dos dados ---
try:
    data = pd.read_csv('processed_data.csv')
    data['Personality'] = data['Personality'].astype(str)

    for col in ['Stage_fear_Yes', 'Drained_after_socializing_Yes']:
        if col in data.columns:
            data[col] = data[col].astype(int)

except FileNotFoundError:
    st.error("Erro: 'processed_data.csv' não encontrado.")
    st.stop()
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

# --- Carregamento do modelo ---
model = None
try:
    model = joblib.load('logistic_regression_model.pkl')
    st.sidebar.success("Modelo carregado com sucesso!")
except FileNotFoundError:
    st.sidebar.warning("Modelo não encontrado: 'logistic_regression_model.pkl'.")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar o modelo: {e}")

# --- Navegação ---
st.sidebar.title("Navegação")
pagina = st.sidebar.selectbox(
    "Selecione a seção:",
    ["Gráfico de Dispersão", "Box/Violin", "Sunburst", "Previsão"]
)

# --- Página: Gráfico de Dispersão ---
if pagina == "Gráfico de Dispersão":
    st.header('Gráfico de Dispersão: Tempo Sozinho vs. Frequência de Postagem')

    fig_scatter = px.scatter(
        data,
        x='Time_spent_Alone',
        y='Post_frequency',
        size='Friends_circle_size',
        color='Personality',
        hover_name=data.index,
        title='Tempo Sozinho vs. Frequência de Postagem',
        height=650
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- Página: Box/Violin ---
elif pagina == "Box/Violin":
    st.header('Distribuição de Variáveis Numéricas por Personalidade')

    numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    target_column = 'Personality'

    selected_numeric_col = st.sidebar.selectbox(
        'Escolha a variável numérica:',
        numeric_columns
    )

    if selected_numeric_col:
        fig_bv = go.Figure()

        fig_bv.add_trace(go.Box(
            y=data[data[target_column] == '0'][selected_numeric_col],
            name='Extrovert (0)'
        ))
        fig_bv.add_trace(go.Box(
            y=data[data[target_column] == '1'][selected_numeric_col],
            name='Introvert (1)'
        ))

        fig_bv.add_trace(go.Violin(
            y=data[data[target_column] == '0'][selected_numeric_col],
            name='Extrovert (0)',
            visible=False
        ))
        fig_bv.add_trace(go.Violin(
            y=data[data[target_column] == '1'][selected_numeric_col],
            name='Introvert (1)',
            visible=False
        ))

        fig_bv.update_layout(
            updatemenus=[
                go.layout.Updatemenu(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"visible": [True, True, False, False]}],
                            label="Box",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [False, False, True, True]}],
                            label="Violin",
                            method="restyle"
                        )
                    ],
                    x=0.1, xanchor="left", y=1.1, yanchor="top"
                )
            ],
            title=f'Distribuição de {selected_numeric_col} por Personalidade',
            yaxis_title=selected_numeric_col,
            height=650
        )
        st.plotly_chart(fig_bv, use_container_width=True)

# --- Página: Sunburst ---
elif pagina == "Sunburst":
    st.header('Gráfico Sunburst por Personalidade')

    try:
        categorized_data = pd.read_csv('categorized_data.csv')
        if categorized_data['Personality'].dtype != 'object':
            categorized_data['Personality'] = categorized_data['Personality'].astype(str)

        available_categorical_columns = [col for col in categorized_data.columns if col.endswith('_category')]

        selected_path_columns = st.sidebar.multiselect(
            'Selecione variáveis para o caminho do Sunburst:',
            available_categorical_columns,
            default=available_categorical_columns
        )

        path_columns = ['Personality'] + selected_path_columns

        if selected_path_columns:
            sunburst_data = categorized_data.groupby(path_columns).size().reset_index(name='count')

            fig_sunburst = px.sunburst(
                sunburst_data,
                path=path_columns,
                values='count',
                title='Distribuição Categórica por Personalidade',
                height=650
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.warning("Selecione pelo menos uma variável além de 'Personality'.")

    except FileNotFoundError:
        st.error("Erro: 'categorized_data.csv' não encontrado.")
    except Exception as e:
        st.error(f"Erro ao criar o gráfico Sunburst: {e}")

# --- Página: Previsão ---
elif pagina == "Previsão":
    st.header('Prever Personalidade')

if model: # Only show this section if the model was loaded successfully
    st.write("Insira os valores para cada característica para prever a personalidade (0: Extrovert, 1: Introvert).")

    # Create input fields for each feature used in the model
    # Get the feature names from the 'data' DataFrame after loading
    # Exclude categorical columns that were not used in training
    feature_names = [col for col in data.columns if col not in ['Personality', 'Stage_fear_Yes', 'Drained_after_socializing_Yes']]


    input_data = {}
    if feature_names: # Only create inputs if feature names are available
        for feature in feature_names:
            # Try to get a default value from the 'data' DataFrame
            default_value = None
            if feature in data.columns:
                if data[feature].dtype in ['float64', 'int64']:
                     default_value = float(data[feature].mean())
                elif data[feature].dtype == 'bool':
                     # Convert boolean to int for mean calculation if needed, but for default value mode is better
                     default_value = bool(data[feature].mode()[0])
                else:
                     default_value = str(data[feature].mode()[0]) # Handle other dtypes appropriately


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
                # Filter feature_names to only include those used in the model
                model_feature_names = [col for col in data.columns if col not in ['Personality', 'Stage_fear_Yes', 'Drained_after_socializing_Yes']]
                input_df = input_df[model_feature_names]


                # Convert boolean columns back to correct type if necessary for prediction
                # Removed the lines that convert boolean columns as they are no longer used in the model
                #for col in ['Stage_fear_Yes', 'Drained_after_socializing_Yes']:
                    #if col in input_df.columns:
                         #input_df[col] = input_df[col].astype(bool) # Convert back to boolean

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

