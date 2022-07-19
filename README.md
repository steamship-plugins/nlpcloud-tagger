# NLPCloud Tagger Plugin - Steamship

This project contains a Steamship Tagger plugin that enables use of NLPCloud's text tagging pipeline.

## Configuration

This plugin must be configured with the following fields:

* `api_key`    - Your One AI API key
* `input_type` - Either `conversation` or `article`. [One AI Documentation](https://studio.oneai.com/docs?api=Pipeline+API&item=Expected+Input+Format&accordion=Introduction%2CPipeline+API%2CNode.js+SDK+Reference%2CClustering+API)
* `skills`     - A CSV list of One AI "skills" that produce tags

Example skills are:

* `entities` - Tags real-world objects, such as people, organizations, and time frames.
* `topics` - Tags text with relevant topics.
* `sentiment` - Tags text with phrases containing positive and negative sentiment: Output tags: `[POS, NEG]`
* `emotions` - Tags text with phrases describing emotion: Output tags: `[happiness, sadness, anger, surprise, fear]`
* `highlights` - Tags text with selected highlights: Output tags: `[highlight]`
* `keywords` - Tags text with selected keywords.
* `sentences` - Splits sentences by text.
* `action_items` - Tags text for action items.

