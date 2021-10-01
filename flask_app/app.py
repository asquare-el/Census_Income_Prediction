from flask import Flask,render_template,request,redirect
from flask import *
import numpy as np
import pickle
app = Flask(__name__)

@app.route("/")
def home():
    # country =[' United-States', ' Cuba', ' Jamaica', ' India', ' ?', ' Mexico',
    #    ' South', ' Puerto-Rico', ' Honduras', ' England', ' Canada',
    #    ' Germany', ' Iran', ' Philippines', ' Italy', ' Poland',
    #    ' Columbia', ' Cambodia', ' Thailand', ' Ecuador', ' Laos',
    #    ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic',
    #    ' El-Salvador', ' France', ' Guatemala', ' China', ' Japan',
    #    ' Yugoslavia', ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Scotland',
    #    ' Trinadad&Tobago', ' Greece', ' Nicaragua', ' Vietnam', ' Hong',
    #    ' Ireland', ' Hungary', ' Holand-Netherlands']
   
    education={' Bachelors': 9,
  ' HS-grad': 11,
  ' 11th': 1,
  ' Masters': 12,
  ' 9th': 6,
  ' Some-college': 15,
  ' Assoc-acdm': 7,
  ' Assoc-voc': 8,
  ' 7th-8th': 5,
  ' Doctorate': 10,
  ' Prof-school': 14,
  ' 5th-6th': 4,
  ' 10th': 0,
  ' 1st-4th': 3,
  ' Preschool': 13,
  ' 12th': 2}
    education=list(education.keys())
    working_class={' State-gov': 7,
  ' Self-emp-not-inc': 6,
  ' Private': 4,
  ' Federal-gov': 1,
  ' Local-gov': 2,
  ' ?': 0,
  ' Self-emp-inc': 5,
  ' Without-pay': 8,
  ' Never-worked': 3}
    working_class=list(working_class.keys())


    occupation=[' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',
       ' Prof-specialty', ' Other-service', ' Sales', ' Craft-repair',
       ' Transport-moving', ' Farming-fishing', ' Machine-op-inspct',
       ' Tech-support', ' ?', ' Protective-serv', ' Armed-Forces',
       ' Priv-house-serv']


    race=[' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo',
       ' Other']
    relationship=[' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried',
       ' Other-relative']
    marital=[' Never-married', ' Married-civ-spouse', ' Divorced',
       ' Married-spouse-absent', ' Separated', ' Married-AF-spouse',
       ' Widowed']
    education_num=[13,  9,  7, 14,  5, 10, 12, 11,  4, 16, 15,  3,  6,  2,  1,  8]
    return render_template("myform.html",education=education,length_education=len(education),race=race,length_race=len(race),relationship=relationship,working_class=working_class,length_working_class=len(working_class),
                    length_relationship=len(relationship),marital=marital,length_marital=len(marital),occupation=occupation,length_occupation=len(occupation),
                    education_num=education_num,length_education_num=len(education_num))

@app.route("/predict",methods=["POST"])
def predict():
    age= request.form['age']
    workclass= request.form["working_class"]
    education= request.form["education"]
    sex= request.form['sex']
    occupation=request.form['occupation']
    edu_number= request.form['education_num']
    marital= request.form['marital']
    capital_gain= request.form['capital_gain']
    capital_loss= request.form['capital_loss']
    hours= request.form['hours']
    fnlwgt= request.form['fnlwgt']
    print(age,workclass,education,sex,occupation,edu_number,marital,capital_loss,capital_gain,hours,fnlwgt)

    working_class={'State-gov': 7,'Self-emp-not-inc': 6,'Private': 4,'Federal-gov': 1,'Local-gov': 2,'?': 0,'Self-emp-inc': 5,'Without-pay': 8,'Never-worked': 3}
    edu={'Bachelors': 9,'HS-grad': 11,'11th': 1,'Masters': 12,'9th': 6, 'Some-college': 15,'Assoc-acdm': 7,'Assoc-voc': 8, '7th-8th': 5, 'Doctorate': 10,'Prof-school': 14,'5th-6th': 4,'10th': 0,'1st-4th': 3, 'Preschool': 13, '12th': 2}
    mar={'Never-married': 4,'Married-civ-spouse': 2,'Divorced': 0,'Married-spouse-absent': 3,'Separated': 5, 'Married-AF-spouse': 1,'Widowed': 6}
    occ={'Adm-clerical': 1,'Exec-managerial': 4,'Handlers-cleaners': 6,'Prof-specialty': 10,'Other-service': 8,'Sales': 12,'Craft-repair': 3,'Transport-moving': 14,'Farming-fishing': 5,'Machine-op-inspct': 7,'Tech-support': 13,'?': 0,'Protective-serv': 11,'Armed-Forces': 2,'Priv-house-serv': 9}
    gender={'Male': 1, 'Female': 0}
    #print(working_class['State-gov'])
    sex=gender[sex]
    workclass=working_class[workclass]
    marital= mar[marital]
    occupation=occ[occupation]
    education= edu[education]
    print(sex)
    print("sex",sex)
    loaded_model = pickle.load(open("..\\randomForestClassifier.pkl", 'rb'))
    inputs=[int(age),workclass,fnlwgt,education,edu_number,marital,occupation,sex,capital_gain,capital_loss,hours]
    val=loaded_model.predict(np.array(inputs).reshape(1,-1))
    print(val[0],type(int(val[0])))
    if int(val[0])==0:
       return "gareeb"
    else:
       return "ameer"
    return "hello"
if __name__=="__main__":
    app.debug=True
    app.run(debug=True)
