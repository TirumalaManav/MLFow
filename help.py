import mlflow

mlflow.set_tracking_uri("http://localhost:5000")


expeiment_id = mlflow.create_experiment("AirPollution Forecasting")

with mlflow.start_run(run_name="AirPollution Forecasting") as run:
    mlflow.set_tag("version", "1.0.0")
    pass

mlflow.end_run()










    # mlflow.log_param("param1", 5)
    # mlflow.log_metric("metric1", 1.0)
    # mlflow.log_metric("metric1", 2.0)
    # mlflow.log_metric("metric1", 3.0)
    # mlflow.log_artifact("data.csv")
    # mlflow.log_artifact("model.pkl")