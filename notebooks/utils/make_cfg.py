import yaml

folds = list(range(5))
for i in folds:
    train_path = []
    for j in range(5):
        if i != j:
            train_path.append(f'./dataset/fold_{j}/images/')

    data = dict(
        train=train_path,
        val=f'./dataset/fold_{i}/images/',
        names=['pbw', 'abw']
    )
    with open(f'/home/mithil/PycharmProjects/Pestedetec2.0/data/fold_{i}.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

