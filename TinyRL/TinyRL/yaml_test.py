import yaml 


with open("gridworld_params.yaml", "r") as cfg:
    try:
        cfg = yaml.safe_load(cfg)
    except yaml.YAMLError as exc:
        print(exc)
print(cfg)