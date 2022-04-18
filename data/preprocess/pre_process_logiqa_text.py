import datasets 

data_path = '../LogiQADataset/'

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

    train = get_dataset(ds_og['train'])
    val = get_dataset(ds_og['validation'])
    test = get_dataset(ds_og['test'])

    dd = datasets.DatasetDict({"train": train, "test": test, "validation": val})

    dd.save_to_disk(data_path + "dataset_text")
    print(dd)


def get_dataset(ds):

  context = []
  endings = []
  question = []
  label = []

  for idx, val in enumerate(ds):
    context.append(val['context'])
    question.append(val['question'])
    endings.append(val['answers'])
    label.append(val['label'])

  return datasets.Dataset.from_dict({"context": context, "question":question, "endings": endings, "label": label})

if __name__ == "__main__":
	main()