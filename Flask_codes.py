#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


#Import libraries 
import numpy as np
from flask import Flask , request , jsonify , render_template
import pickle


# # Flask Application

# In[2]:


#Create Flask aapplication 
app = Flask(__name__)

#Load the Pickle model
model =  pickle.load(open('pred_model.pkl', 'rb'))


# In[ ]:


#Create Home Page

@app.route('/', endpoint = 'Home1')
#Create function for home 
def Home1():
    return render_template('index.html' )

#Define Predict method
#Since we want to recieve data,we use POST method

@app.route('/predict' , methods = ['POST'])
def predict():
    float_features = [int(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    
    return render_template('index.html' , prediction_text = 'The value is {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug= True, use_reloader=False)
    


# In[ ]:





# In[ ]:




