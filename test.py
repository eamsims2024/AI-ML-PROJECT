import mlflow
from mlflow.tracking import MlflowClient
import mlflow.spacy
import dagshub

# Initialize DagsHub
dagshub.init(repo_owner='Samcool1990', repo_name='mlflow_spacy_proj', mlflow=True, root='./Artifacts')

model_name = "spacy_combined_ner_model_stanza_test_experiment_model_11"
# Optional: Set the MLflow tracking URI if you are using a remote server
model_uri = f"models:/{model_name}/5" 
mlflow.set_tracking_uri(model_uri)

# List runs in the experiment to find the correct run ID
experiment_id = "spacy_stanza_combined_model_experiment_11"  # Replace with your experiment ID
client = MlflowClient()
runs = client.search_runs(experiment_ids=[experiment_id])

for run in runs:
    print(f"Run ID: {run.run_id}")

# Replace with the correct run ID found from the above output
correct_run_id = "cc257ff14028403da2002a3630119bc3"

# Load model using the correct run ID
logged_model = f'runs:/{correct_run_id}/spacy_combined_model_experiment_11'
loaded_model = mlflow.spacy.load_model(logged_model)
