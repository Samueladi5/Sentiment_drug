import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns


# Load data
data = pd.read_csv('drugsComTrain_raw.csv')

# Variable untuk menyimpan data kondisi dari Halaman 1
selected_condition = None

# Elemen dummy untuk memaksa Streamlit merender ulang konten
dummy_rendezvous = st.empty()

# Page 1
def recommend_drugs():
    global selected_condition

    st.title('Drug Recommendation Sentiment Predictor')

    # User input for medical condition
    condition = st.text_input('Masukkan kondisi medis:')

    if st.button('Submit'):
        if condition:
            # Filter data based on the specified medical condition
            condition_data = data[data['condition'] == condition]

            # Check if there are enough data for visualization
            if len(condition_data) >= 5:
                # Calculate the number of occurrences of each drug in the filtered data
                drug_counts = condition_data['drugName'].value_counts().reset_index()
                drug_counts.columns = ['drugName', 'count']

                # Filter drugs that have appeared at least 5 times
                popular_drugs = drug_counts[drug_counts['count'] >= 5]

                # Calculate average rating for each drug in the given condition
                avg_ratings = condition_data[condition_data['drugName'].isin(popular_drugs['drugName'])].groupby('drugName')['rating'].mean().reset_index()

                # Sort by average rating in descending order and get the top 10
                recommended_drugs = avg_ratings.sort_values(by='rating', ascending=False).head(10)

                # Display the recommended drugs table
                st.table(recommended_drugs[['drugName', 'rating']])

                # Visualization: Bar plot of average ratings
                plt.figure(figsize=(10, 6))
                sns.barplot(x='drugName', y='rating', data=recommended_drugs)
                plt.xticks(rotation=45, ha='right')

                # Display the plot using Streamlit
                st.pyplot(plt.gcf())

                # Update the selected condition
                selected_condition = condition

                # Force Streamlit to rerender content
                dummy_rendezvous.text(condition)

            else:
                st.warning(f"Tidak cukup data untuk visualisasi kondisi '{condition}'.")

        else:
            st.warning('Masukkan kondisi medis untuk mendapatkan rekomendasi obat.')

# Page 2
def visualize_data():
    global selected_condition

    st.title('Visualisasi Data')

    # User input for drug name and condition
    drug_name = st.text_input('Masukkan nama obat:')
    condition_input = st.text_input('Masukkan kondisi medis:')

    if st.button('Submit'):
        if drug_name and condition_input:
            # Filter data based on the specified drug name and condition
            filtered_data = data[(data['drugName'] == drug_name) & (data['condition'] == condition_input)]

            if not filtered_data.empty:
                # WordCloud
                st.subheader('WordCloud')
                wordcloud_text = ' '.join(filtered_data['review'].dropna().astype(str))
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
                st.image(wordcloud.to_image(), caption='WordCloud')

                # Display table of reviews
                st.subheader('Tabel Review')
                # Set the width and height of the DataFrame table
                table_width = 800
                table_height = 400
                st.dataframe(filtered_data[['review', 'rating']], width=table_width, height=table_height)

            else:
                st.warning(f"Tidak ada data untuk obat '{drug_name}' pada kondisi medis '{condition_input}'.")

        else:
            st.warning('Masukkan nama obat dan kondisi medis untuk visualisasi data.')

# Choose page using sidebar
page = st.sidebar.selectbox("Pilih Halaman", ["Rekomendasi Obat", "Visualisasi Data"])

# Display the selected page
if page == "Rekomendasi Obat":
    recommend_drugs()
elif page == "Visualisasi Data":
    visualize_data()
