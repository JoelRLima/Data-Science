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
    st.header('Prever Personalidade com o Modelo')

    if model:
        feature_names = [col for col in data.columns if col != 'Personality']
        input_data = {}

        for feature in feature_names:
            default = float(data[feature].mean()) if data[feature].dtype in ['float64', 'int64'] else data[feature].mode()[0]

            if data[feature].dtype in ['float64', 'int64']:
                input_data[feature] = st.number_input(f'{feature}:', value=default)
            elif data[feature].dtype == 'bool':
                input_data[feature] = st.checkbox(f'{feature}:', value=bool(default))
            else:
                input_data[feature] = st.text_input(f'{feature}:', value=str(default))

        if st.button("Prever"):
            try:
                input_df = pd.DataFrame([input_data])[feature_names]

                for col in ['Stage_fear_Yes', 'Drained_after_socializing_Yes']:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype(bool)

                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]

                resultado = "Introvertido" if pred == 1 else "Extrovertido"
                st.subheader(f"Resultado: **{resultado}**")
                st.write(f"Probabilidade (Extrovertido): {proba[0]:.4f}")
                st.write(f"Probabilidade (Introvertido): {proba[1]:.4f}")

            except Exception as e:
                st.error(f"Erro durante a predição: {e}")
    else:
        st.warning("Modelo não carregado. A previsão não está disponível.")
