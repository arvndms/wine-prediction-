from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
model = pickle.load(open("model/model.pkl","rb"))
scaler = pickle.load(open("model/scaler.pkl","rb"))
ohe = pickle.load(open("model/ohe.pkl","rb"))
df = pd.read_csv('data/beer-servings.csv')
beer_avg = round(df['beer_servings'].mean(),2)
wine_avg = round(df['wine_servings'].mean(),2)
spirit_avg = round(df['spirit_servings'].mean(),2)

@app.route('/')
def home():
    return render_template('home.html',avg_beer=beer_avg,avg_spirit=spirit_avg,avg_wine=wine_avg )

@app.route('/predict',methods=['POST'])
def predict():
    wine=float(request.values['Wine'])
    beer=float(request.values['Beer'])
    spirit=float(request.values['Spirit'])
    continent = request.values['continent']
    #encode continent
    encoded = ohe.transform([[continent]]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['continent']))
    #scaling 
    num_df = pd.DataFrame({
        'beer_servings':[beer],
        "spirit_servings":[spirit],
        "wine_servings":[wine]})
    final_df = pd.concat([num_df, encoded_df], axis=1)
    final_df = final_df[scaler.feature_names_in_]
    scaled = scaler.transform(final_df)
    #prediction
    prediction = model.predict(scaled)

    return render_template('res.html',prediction=round(prediction[0],2))

if __name__ == "__main__":
    app.run(debug=True,port=8000)