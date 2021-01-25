import json

#load template json
with open(snakemake.input.template_json) as f:
    dataset = json.load(f)

dataset['training'] = [{'image': img, 'label': lbl} for img,lbl in zip(
    snakemake.input.training_imgs,
    snakemake.input.training_lbls)]

dataset['modality'] = [{num: name} for num,name in zip(
    [str(i).zfill(4) for i in range(0,len(snakemake.params.modalities))],
    snakemake.params.modalities)]

dataset['numTraining'] = len(dataset['training'])

#write modified json
with open(snakemake.output.dataset_json, 'w') as f:
    json.dump(dataset, f, indent=4)
