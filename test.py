from src.models import load_intervenable_model

model = load_intervenable_model("models/pythia-14m")
print(model.run_intervention)