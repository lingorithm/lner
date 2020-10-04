# Lingorithm Named Entity Recognition
This package is the core function for any NLP operation or pacakge used by lingorithm.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install lner.

```bash
pip install lner
```

# Usage
### Data 
Input data is a list of dict 
```Python
entities = {
    'countries': [ 
        {values: ['Afghanistan'], key: 'AF'}, 
        {values: ['Åland Islands'], key: 'AX'}, 
        ....
        {values: ['Zimbabwe'], key: 'ZW'} 
    ],
    "colors": [
        {
        "key": "black",
        "values": ["black"],
        "meta": {
            "rgba": [255,255,255,1],
            "hex": "#000"
        }
        },
       ...
        "meta": {
            "rgba": [0,255,0,1],
            "hex": "#0F0"
        }
        },
    ]
}


sentences = [
    ('Egypt\'s flag containes black, white and red.', [
        (0, 5, 'countries'),
        (23, 28, 'colors'),
        (40, 43, 'colors')
    ])
    .....
]

```
```Python
import lner

ner = lner('en', entities, sentences) 

ner.process()

print(ner.data)
# list of Tuples
# [
#     (['Green', ',', 'yellow', 'are', 'the', 'main', 'colors', 'for', 'Zimbabwe', '\'s' 'flag','.'],
#     ['B-colors', 'O', 'B-colors', 'O', 'O', 'O', 'O', 'O', 'B-countries', 'O', 'O', 'O'])
# ]

ner.train(epochs=20, batch_size=8, model_name="countries_colors_v1")


entities, tokens = ner.recognize("yellow is the main colors for Northern Mariana Islands flag.")
print(entities)
# [
#     [
#         {
#             'key': 'yellow', 
#             'meta': {
#                 'rgba': [
#                     255,
#                     255,
#                     0,
#                     1
#                 ],
#                 'hex': '#FF0'
#             }, 
#             'value': 'yellow', 
#             'entity': 'colors'
#         },
#         {
#             'key': 'MP', 
#             'meta': None, 
#             'value': 'Northern Mariana Islands', 
#             'entity': 'countries'
#         }
#     ]
# ]
print(tokens) 
# [['yellow', 'is', 'the', 'main', 'colors', 'for', 'Northern', 'Mariana', 'Islands', 'flag', '.']]
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)