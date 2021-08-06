from flask import Flask,render_template,request,jsonify
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def home():
	return render_template('home.html',value=2)


@app.route("/result",methods=['GET','POST'])
def result():
	textarea= request.form.get('textarea')

	data=pd.read_csv('./spam.csv')
	data.groupby('Category').describe()
	data['spam']=data['Category'].apply(lambda x: 1 if x=='spam' else 0)
	X_train,X_test,Y_train,Y_test=train_test_split(data.Message,data.spam,test_size=0.25)
	vectorizer = CountVectorizer()
	X_train_count=vectorizer.fit_transform(X_train.values)
	model=MultinomialNB()
	model.fit(X_train_count,Y_train)
	data=[]
	data.append(textarea)
	print(data)
	text_count=vectorizer.transform(data)
	return render_template('home.html', value=model.predict(text_count))



if __name__=='__main__':
	app.run()