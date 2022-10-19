Fine-Tuning fusion model for case-hold/logiqa dataset:

Setup for this project
- put pre-processed data into data folder
    - one dataset for text and one dataset for AMR data (generated by Spring AMR parser)
- put AMRBART model into amrbart folder (https://github.com/goodbai-nlp/AMRBART)
- update paths in scripts/run*.sh files

Other information:
- Spring is part of the project since we need the PENMANBartTokenizer (which is used by AMRBART).
- The fusion model fine-tuned on CaseHOLD can be found here: https://huggingface.co/niks883/fusion-case-hold
- The fusion model fine-tuned on LogiQA can be found here: https://huggingface.co/niks883/fusion-logiqa