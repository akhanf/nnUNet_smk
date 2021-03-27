import json

#load template json
with open(snakemake.input.template_json) as f:
    dataset = json.load(f)

dataset['training'] = [{'image': img, 'label': lbl} for img,lbl in zip(
    snakemake.params.training_imgs_nosuffix,
    snakemake.input.training_lbls)]

t = snakemake.params.modalities
dataset['modality'] = {x: y for x,y in zip(range(len(t)),t)}

dataset['numTraining'] = len(dataset['training'])

#write modified json
with open(snakemake.output.dataset_json, 'w') as f:
    json.dump(dataset, f, indent=4)

# import json
# with open('template.json') as f:
#     dataset = json.load(f)
# t = ['T1w', 'T2w']
