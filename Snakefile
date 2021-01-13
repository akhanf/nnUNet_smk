
#snakefile to re-factor input data into format that nnUNet likes
configfile: 'config.yml'

#get list of all subj ids
Lsubjids, = glob_wildcards(config['in_image'].format(hemi='L',subjid='{subjid}'))
Rsubjids, = glob_wildcards(config['in_image'].format(hemi='R',subjid='{subjid}'))

#only take subjects that have both left and right (to make it easier, and balanced)
subjids = set(Lsubjids).intersection(set(Rsubjids))
print(f'number of subjects with both left and right: {len(subjids)}')

#do 80/20 split of training/test
import random
random.seed(0)
num_training = int(0.8 * len(subjids))
shuffle_subjids = random.sample(subjids,k=len(subjids))
training_subjids = shuffle_subjids[:num_training]
testing_subjids = shuffle_subjids[num_training:]

hemis=['L','R']

print(f'number of training subjects: {len(training_subjids)}')
print(f'number of test subjects: {len(testing_subjids)}')


localrules: resample_training_img,resample_training_lbl,plan_preprocess,create_dataset_json

rule all_predict:
    input:
        testing_imgs = expand('raw_data/nnUNet_predictions/{arch}/{task}/{trainer}__nnUNetPlansv2.1/{checkpoint}/hcp_{subjid}{hemi}.nii.gz',subjid=testing_subjids, hemi=hemis, arch=config['architecture'], task=config['task'], trainer=config['trainer'],checkpoint=config['checkpoint'],allow_missing=True),
 

rule all_train:
    input:
       expand('trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_final_checkpoint.model',fold=range(5), arch=config['architecture'], task=config['task'], trainer=config['trainer'])

       

rule resample_training_img:
    input: config['in_image']
    params:
        resample_res = config['out_resolution']
    output: 'raw_data/nnUNet_raw_data/{task}/imagesTr/hcp_{subjid}{hemi}_0000.nii.gz'
    threads: 32 #to make it serial on a node
    group: 'preproc'
    shell: 'c3d  {input} -resample {params.resample_res} -o {output}'

rule resample_testing_img:
    input: config['in_image']
    params:
        resample_res = config['out_resolution']
    output: 'raw_data/nnUNet_raw_data/{task}/imagesTs/hcp_{subjid}{hemi}_0000.nii.gz'
    group: 'preproc'
    threads: 32 #to make it serial on a node
    shell: 'c3d  {input} -resample {params.resample_res} -o {output}'


rule resample_training_lbl:
    input: config['in_label']
    params:
        resample_res = config['out_resolution']
    output: 'raw_data/nnUNet_raw_data/{task}/labelsTr/hcp_{subjid}{hemi}.nii.gz'
    group: 'preproc'
    threads: 32 #to make it serial on a node
    shell: 'c3d {input} -interpolation NearestNeighbor -resample {params.resample_res} -o {output}'


rule create_dataset_json:
    input: 
        training_imgs = expand('raw_data/nnUNet_raw_data/{task}/imagesTr/hcp_{subjid}{hemi}_0000.nii.gz',subjid=training_subjids, hemi=hemis, allow_missing=True),
        training_lbls = expand('raw_data/nnUNet_raw_data/{task}/labelsTr/hcp_{subjid}{hemi}.nii.gz',subjid=training_subjids, hemi=hemis, allow_missing=True),
        template_json = 'template.json'
    params:
        training_imgs_nosuffix = expand('raw_data/nnUNet_raw_data/{task}/imagesTr/hcp_{subjid}{hemi}.nii.gz',subjid=training_subjids, hemi=hemis, allow_missing=True),
    output: 
        dataset_json = 'raw_data/nnUNet_raw_data/{task}/dataset.json'
    group: 'preproc'
    script: 'create_json.py' 
     
 
rule plan_preprocess:
    input: 
        dataset_json = 'raw_data/nnUNet_raw_data/{task}/dataset.json'
    params:
        env = '0_setenv.sh',
        venv = '../venv/bin/activate',
        task_num = lambda wildcards: re.search('Task([0-9]+)\w*',wildcards.task).group(1),
    output: 
        dataset_json = 'preprocessed/{task}/dataset.json'
    group: 'preproc'
    resources:
        threads = 8,
        mem_mb = 16000
    shell:
        'source {params.env} && source {params.venv} && '
        'nnUNet_plan_and_preprocess  -t {params.task_num} --verify_dataset_integrity'

def get_checkpoint_opt(wildcards, output):
    if os.path.exists(output.latest_model):
        return '--continue_training'
    else:
        return '' 
      
rule train_fold:
    input:
        dataset_json = 'preprocessed/{task}/dataset.json'
    params:
        env = '0_setenv_localscratch.sh',
        venv = '../venv/bin/activate',
        #add --continue_training option if a checkpoint exists
        checkpoint_opt = get_checkpoint_opt
    output:
        final_model = 'trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_final_checkpoint.model',
        latest_model = 'trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_latest.model',
        best_model = 'trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/model_best.model'
    threads: 16
    resources:
        gpus = 1,
        mem_mb = 32000,
        time = 1440,
    group: 'train'
    shell:
        'source {params.env} && source {params.venv} && '
        'rsync -av preprocessed $TMPDIR && '
        'nnUNet_train  {params.checkpoint_opt} {wildcards.arch} {wildcards.trainer} {wildcards.task} {wildcards.fold}'




rule predict_test_subj_with_latest:
    input:
        latest_model = expand('trained_models/nnUNet/{arch}/{task}/{trainer}__nnUNetPlansv2.1/fold_{fold}/{checkpoint}.model',fold=range(5),allow_missing=True),
        testing_imgs = expand('raw_data/nnUNet_raw_data/{task}/imagesTs/hcp_{subjid}{hemi}_0000.nii.gz',subjid=testing_subjids, hemi=hemis, allow_missing=True),
    params:
        in_folder = 'raw_data/nnUNet_raw_data/{task}/imagesTs',
        out_folder = 'raw_data/nnUNet_predictions/{arch}/{task}/{trainer}__nnUNetPlansv2.1/{checkpoint}',
        env = '0_setenv.sh',
        venv = '../venv/bin/activate',
    output:
        testing_imgs = expand('raw_data/nnUNet_predictions/{arch}/{task}/{trainer}__nnUNetPlansv2.1/{checkpoint}/hcp_{subjid}{hemi}.nii.gz',subjid=testing_subjids, hemi=hemis, allow_missing=True),
    threads: 8 
    resources:
        gpus = 1,
        mem_mb = 32000,
        time = 30,
    group: 'inference'
    shell:
        'source {params.env} && source {params.venv} && '
        'nnUNet_predict  -chk {wildcards.checkpoint}  -i {params.in_folder} -o {params.out_folder} -t {wildcards.task}'

   

