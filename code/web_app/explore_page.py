import streamlit as st
import pandas as pd

# Load dataset from the correct path
@st.cache_data
def load_data():
    df = pd.read_excel("data/Heart Disease Dataset 4.xlsx")  # Corrected file path
    return df

df = load_data()


# Define the function properly
def show_explore_page():
    st.title("Explore Heart Disease Dataset")

    st.write("""### Data Distribution""")

    # # Feature Distribution Plot
    # feature = st.selectbox("Select a Feature", df.columns)
    # fig, ax = plt.subplots()
    # sns.histplot(df[feature], bins=30, kde=True, ax=ax)
    # st.pyplot(fig)

    # # Correlation Heatmap
    # st.write("""### Correlation Between Features""")
    # fig, ax = plt.subplots(figsize=(10, 6))
    # sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    # st.pyplot(fig)

    # Mean Heart Disease Risk Based on Age
    st.write("""### Heart Disease Risk by Age""")
    data = df.groupby("age")["num"].mean()
    st.line_chart(data)
