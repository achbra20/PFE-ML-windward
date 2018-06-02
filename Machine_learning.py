from flask import Flask, url_for, redirect, session, flash
from apscheduler.schedulers.background import BackgroundScheduler
from flask import jsonify
from flask import request ,render_template
import os
import datetime
from Ml_destination        import Creation_Model,Predection_distination ,Training_model_destination,Creation_Model_recommendation ,Recommendation_distnation
from ML_boats              import Create_Data_Recommendation_Boats,Recommendation_boats,ML_boat_data,Prediction,Training_boat
from  ML_boats_destination import Create_data_Ml_boat_destination,Ml_boat_destination,Recommendation_boats_destenation
from Dashbord              import Statestique_all_destination,Statestique_bosts


app = Flask(__name__)
app.config.from_pyfile("myconfig.cfg")

################tache cron#################
def sensor():
    """ Function for test purposes. """
    print(" hello tache cron")

sched = BackgroundScheduler(daemon=True)
sched.add_job(sensor,'interval',minutes=1)
#sched.start()



##########################################
@app.route('/predict')
def predict():
    root_model = app.config['ROOT_MODEL']
    name_model = app.config['NAME_MODEL_DESTINATION']
    predection = Predection_distination(root_model+"/"+name_model)
    print(id(predection))
    x= predection.predict()
    return 'helo'
##########################################

@app.route('/', methods=['GET','POST'])
def login():
    login = app.config['LOGIN_ML']
    pwd   = app.config['PASSWORD_ML']
    response = ""
    if request.method == "POST":
        if request.form['login'] == login and request.form['pwd'] == pwd:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            response= "Invalid connexion"
    return render_template('Login.html', response = response)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out.')
    return render_template('Login.html')

@app.route('/index')
def index():
    x =0
    if  not session.get('logged_in'): #x == 1:#
        return redirect(url_for('login'))
    else:
        model_name_ml       = app.config['NAME_MODEL_DESTINATION']
        model_ml_boat_des   = app.config['NAME_MODEL_BOATS_DESTINATION']
        model_ml_boat       = app.config['NAME_MODEL_BOATS']
        root_model          = app.config['ROOT_MODEL']
        api_all_destination = app.config['API_ALL_DESTINATION']
        api_boats_generic   = app.config['API_ALL_BOATS_ID_WW']

        statestique         = Statestique_all_destination()
        #scored_destination  = statestique.scored_day(model_name_ml, root_model,api_all_destination)
        score_train ,score_test,error_train,error_test = statestique.score_ml_destination(root_model)
        all_destination     = statestique.found_tuple(root_model,api_all_destination)
        countrys            = statestique.all_countries()
        all_boats           = Statestique_bosts.all_boats(api_boats_generic)

        boats_score                 = Statestique_bosts.top_boats(api_boats_generic,root_model)
        all_destination_score       = Statestique_all_destination.top_destination_request(root_model,api_all_destination)
        all_des_boats,all_boats_des = Statestique_bosts.all_boats_destination(root_model,api_boats_generic,api_all_destination)

        creation_model_destination = datetime.datetime.fromtimestamp(os.path.getmtime(root_model + "/" + model_name_ml))
        creation_rec_destination   = datetime.datetime.fromtimestamp(os.path.getmtime(root_model + "/" + "recommendation_destination.json"))
        creation_data_destination  = datetime.datetime.fromtimestamp(os.path.getmtime(root_model + "/" + "data_Ml_destination.json"))
        creation_rec_boats         = datetime.datetime.fromtimestamp(os.path.getmtime(root_model + "/" + "data_final_boats_recommendation.json"))
        creation_data_boat_des     = datetime.datetime.fromtimestamp(os.path.getmtime(root_model + "/" + "data_final_boats_destination.json"))
        creation_data_boat         = datetime.datetime.fromtimestamp(os.path.getmtime(root_model + "/" + "data_Ml_boat.json"))
        creation_model_boat_des    = datetime.datetime.fromtimestamp(os.path.getmtime(root_model + "/" + model_ml_boat_des))
        creation_model_boat        = datetime.datetime.fromtimestamp(os.path.getmtime(root_model + "/" + model_ml_boat))

        return render_template('index.html',creation_model_boat= creation_model_boat ,creation_data_boat =creation_data_boat,all_destination_score= all_destination_score,boats_score=boats_score,all_des_boats = all_des_boats,all_boats_des =all_boats_des,date_model_boat_des= creation_model_boat_des,all_boats =all_boats ,date_creation_boast_des= creation_data_boat_des,creation_rec_boats =creation_rec_boats,countrys = countrys,creation_data_destination=creation_data_destination,creation_rec_destination=creation_rec_destination,date_model_destination = creation_model_destination,all_destination = all_destination, error_ml = error_train,score = score_test ,date = datetime.datetime.now().strftime('%d/%m/%Y'))

@app.route('/graphe/<country>', methods=['GET'])
def build_graph(country):
    root_model = app.config['ROOT_MODEL']
    statestique = Statestique_all_destination()
    destination_statistique = statestique.courbe_destination(root_model, country)

    return jsonify(destination_statistique)

@app.route('/graphe_boat/<id_gen>')
def build_graphe_boat(id_gen):
    root_model = app.config['ROOT_MODEL']
    statistique = Statestique_bosts()
    boat_stestique = statistique.satestique_boat_generique(id_gen,root_model)

    return jsonify(boat_stestique)

@app.route('/graphe_boat_des/<id_gen>/<id_location>')
def build_graphe_boat_des(id_gen,id_location):
    root_model = app.config['ROOT_MODEL']
    statistique = Statestique_bosts()
    boat_stestique = statistique.satestique_boat_generique_destination(id_gen,id_location,root_model)

    return jsonify(boat_stestique)

@app.route('/graphe_type_boat_des/<id_location>')
def build_graphe_type_boat_des(id_location):
    root_model   = app.config['ROOT_MODEL']
    boat_generic = app.config['API_ALL_BOATS_GENERIC']
    statistique = Statestique_bosts()
    boat_stestique = statistique.boat_type_destination(root_model,id_location,boat_generic)

    return jsonify(boat_stestique)

@app.route('/graphe_destination_boat/<id_location>')
def build_graphe_destination_boat(id_location):
    root_model   = app.config['ROOT_MODEL']
    api_all_boats = app.config['API_ALL_BOATS_ID_WW']
    boat_stestique = Statestique_bosts.destination_boat_statetique(root_model,id_location,api_all_boats)

    return jsonify(boat_stestique)

@app.route('/type_boat_country/<country>')
def build_graphe_country_boat(country):
    api_ww               = app.config['API_REQUEST_MONTH_WW']
    api_crm              = app.config['API_REQUEST_MONTH_CRM']
    statistique = Statestique_bosts()
    botas = statistique.statestique_type_boat(api_ww,api_crm,country)
    return jsonify(botas)


#################### Recommendation destination#######################

@app.route('/update_data')
def update_data():
    api_ww              = app.config['API_ALL_REQUEST_DATE_WW']
    api_crm             = app.config['API_ALL_REQUEST_DATE_CRM']
    api_all_destination = app.config['API_ALL_DESTINATION']
    root_model          = app.config['ROOT_MODEL']
    model = Creation_Model(api_ww,api_crm,api_all_destination,root_model)
    try:
        model.create_model()
    except ValueError:
        print(ValueError)
        jsonify(result= "Oops! There is a probleme in the creation of the dataframe of the training")

    return jsonify(result='Job done :) ')

@app.route('/new_training_model_destination')
def training_model_destination():
    path       = app.config['ROOT_MODEL']
    name_model = app.config['NAME_MODEL_DESTINATION']
    model      =  Training_model_destination(path,name_model)
    try:
        model.training()
    except ValueError:
        return jsonify(result="Oops! There is a probleme in the training")
    return jsonify(result='Job done :) ')

@app.route('/create_data_recommendation/')
def recommendation_model():
    api_ww               = app.config['API_REQUEST_MONTH_WW']
    api_crm              = app.config['API_REQUEST_MONTH_CRM']
    api_all_destination  = app.config['API_ALL_DESTINATION']
    root_model           = app.config['ROOT_MODEL']
    recommendation       = Creation_Model_recommendation(api_ww,api_crm,api_all_destination,root_model)
    try:
         model_recommendation = recommendation.create_m_recomendation()
    except ValueError:
        print("There is some error in the building of dataframe of recommmendation")
    return jsonify(result='Job done :) ')

@app.route('/rec_desination/<country>/<date>/<nb_dest>')
def recommendation(country,date,nb_dest):
    root_model        = app.config['ROOT_MODEL']
    api_all_des       = app.config['API_ALL_DESTINATION']
    model_name_ml     = app.config['NAME_MODEL_DESTINATION']
    ml_recommendation = Recommendation_distnation(root_model)
    destination = ml_recommendation.list_recommended(country, date, nb_dest, api_all_des, model_name_ml)
    try:
        print('rrrr')
    except ValueError:
        print("There is some error")
        return jsonify(result='There is some error')
    return jsonify({'destinations': destination})

#################### Recommendation boats    ########################

@app.route('/create_data_recommendation_boats')
def create_data_recommendation_boats():
    api_ww               = app.config['API_REQUEST_MONTH_WW']
    api_crm              = app.config['API_REQUEST_MONTH_CRM_BOAT']
    api_boats_all        = app.config['API_ALL_BOATS_ID_WW']
    root_model           = app.config['ROOT_MODEL']
    create_data_rec = Create_Data_Recommendation_Boats(api_ww,api_crm,api_boats_all)
    try:
        create_data_rec.created_data_for_recommendation_boat(root_model)
    except ValueError:
        print("There is some error in the building of dataframe of recommmendation")
        return jsonify(result='There is some error')
    return jsonify(result='Job done :) ')

@app.route('/recommendation_boats/<country>/<date>/<nombre>')
def recommendation_boats(country,date,nombre):
    root_model = app.config['ROOT_MODEL']
    name_model = app.config['NAME_MODEL_BOATS']
    recommendation_boats_country = Recommendation_boats(root_model)
    try:
        boats_recommendation = recommendation_boats_country.recommandation_boat(country, date, name_model, root_model,nombre)
    except ValueError:
        print("There is some error in  recommmandation")
        return jsonify(result='Erreur :( ')
    return jsonify({'boats':boats_recommendation})

@app.route('/create_data_boats')
def create_data_boat():
    api_boats_ww   = app.config['API_ALL_REQUEST_DATE_WW']
    api_boat_crm   = app.config['API_ALL_REQUEST_DATE_CRM']
    api_boats_crm2 = app.config['API_ALL_REQUEST_BOAT_CRM']
    api_boat_id    = app.config['API_ALL_BOATS_ID_WW']
    root_model     = app.config['ROOT_MODEL']
    new_data       = ML_boat_data(api_boats_ww,api_boat_crm,api_boats_crm2,api_boat_id)
    try:
        new_data.create_data_ml_boats(root_model)
    except ValueError:
        print("There is some error in the building of dataframe of recommmendation")
        return jsonify(result='Erreur :( ')
    return jsonify(result='Job done :) ')

@app.route('/training_data_boats')
def training_data_boat():

    root_model = app.config['ROOT_MODEL']
    name_model = app.config['NAME_MODEL_BOATS']
    try:
        training_boats = Training_boat(root_model,name_model)
    except ValueError:
        print("There is some error in the building of data")
        return jsonify(result='There is some error')
    return jsonify(result='Job done :) ')


#################### Recommendation boat_destination ###############

@app.route('/create_data_boats_destination')
def create_data_recommendation_boats_destination():
    api_ww               = app.config['API_ALL_REQUEST_DATE_WW']
    api_crm              = app.config['API_ALL_REQUEST_DATE_CRM']
    api_crm2              = app.config['API_BOATS_REQUEST_DATE_CRM']
    api_boats_all        = app.config['API_ALL_BOATS_ID_WW']
    api_boats_generic    = app.config['API_ALL_BOATS_GENERIC']
    api_all_destination  = app.config['API_ALL_DESTINATION']
    root_model           = app.config['ROOT_MODEL']
    create_data_rec = Create_data_Ml_boat_destination(api_ww ,api_crm,api_crm2 ,api_boats_all ,api_boats_generic,api_all_destination)
    try:
        create_data_rec.create_data_ML(root_model)
    except ValueError:
        print("There is some error in the building of data")
        return jsonify(result='There is some error')
    return jsonify(result='Job done :) ')

@app.route('/training_model_boats_destination')
def training_model_boats_destination():
    root_model = app.config['ROOT_MODEL']
    name_model = app.config['NAME_MODEL_BOATS_DESTINATION']
    try:
        training_boats_des = Ml_boat_destination(root_model,name_model)
    except ValueError:
        print("There is some error in the building of data")
        return jsonify(result='There is some error')
    return jsonify(result='Job done :) ')

@app.route("/recommendation_boat_destination/<country>/<id_location>/<date>")
def prediction_boats_destination(country,id_location,date):
    root_model           = app.config['ROOT_MODEL']
    api_boats_generic    = app.config['API_ALL_BOATS_GENERIC']
    api_all_destination  = app.config['API_ALL_DESTINATION']
    name_model           = app.config['NAME_MODEL_BOATS_DESTINATION']
    recommendation_boats_destination = Recommendation_boats_destenation(api_boats_generic,api_all_destination,root_model,name_model)
    try:
        boats = recommendation_boats_destination.scored_boat(country, id_location, date)
    except ValueError:
        print("There is some error in the building of data")
        return jsonify(result='There is some error')
    return jsonify({'boats':boats})


#######################################

@app.route('/begin_all')
def begin_all():
    update_data()
    recommendation_model()
    create_data_recommendation_boats()
    create_data_boat()


#######################################

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(host = '40.89.130.89',port=5000,threaded=True)



