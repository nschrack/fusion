import datasets 

data_path = '../LogiQADataset/'

train_file_context = data_path+ '/preprocessed/context/train.jsonl'
val_file_context =  data_path+ '/preprocessed/context/val.jsonl'
test_file_context =  data_path+ '/preprocessed/context/test.jsonl'

train_file_endings =  data_path+ '/preprocessed/endings/train.jsonl'
val_file_endings = data_path+ '/preprocessed/endings/val.jsonl'
test_file_endings = data_path+ '/preprocessed/endings/test.jsonl'

train_file_question  =  data_path+ '/preprocessed/question/train.jsonl'
val_file_question = data_path+ '/preprocessed/question/val.jsonl'
test_file_question  = data_path+ '/preprocessed/question/test.jsonl'

train_file ='/Users/niko/ML/parse_amr/LogiQADataset/train.jsonl'
val_file =  '/Users/niko/ML/parse_amr/LogiQADataset/val.jsonl'
test_file = '/Users/niko/ML/parse_amr/LogiQADataset/test.jsonl'

number_of_answers = 4

def main():

    data_files = {}
    data_files['test'] = test_file
    data_files['validation'] = val_file
    data_files['train'] = train_file

    ds_og = datasets.load_dataset(
        "json", data_files=data_files,
    )

    data_files = {}
    data_files['test'] = test_file_context
    data_files['validation'] = val_file_context
    data_files['train'] = train_file_context

    ds_context = datasets.load_dataset(
    "json", data_files=data_files,
    )

    data_files = {}
    data_files['test'] = test_file_endings
    data_files['validation'] = val_file_endings
    data_files['train'] = train_file_endings

    ds_endings = datasets.load_dataset(
    "json", data_files=data_files,
    )

    data_files = {}
    data_files['test'] = test_file_question
    data_files['validation'] = val_file_question
    data_files['train'] = train_file_question

    ds_question = datasets.load_dataset(
    "json", data_files=data_files,
    )

    train = get_dataset(ds_og['train'], ds_context['train'], ds_endings['train'], ds_question['train'])
    val = get_dataset(ds_og['validation'], ds_context['validation'], ds_endings['validation'], ds_question['validation'])
    test = get_dataset(ds_og['test'], ds_context['test'], ds_endings['test'], ds_question['test'])

    dd = datasets.DatasetDict({"train": train, "test": test, "validation": val})

    dd.save_to_disk(data_path + "dataset_amr")
    print(dd)



def get_dataset(ds, ds_context, ds_endings, ds_question):
  ds_endings_packed = [ds_endings[i:i+number_of_answers] for i in range(0,len(ds_endings),number_of_answers)]

  context = []
  endings = []
  question = []
  label = []

  for idx, val in enumerate(ds):
    context.append(ds_context[idx]['tgt'])
    question.append(ds_question[idx]['tgt'])
    endings.append(ds_endings_packed[idx]['tgt'])
    label.append(val['label'])

  return datasets.Dataset.from_dict({"context": context, "question":question, "endings": endings, "label": label})

if __name__ == "__main__":
	main()