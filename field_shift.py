GENERIC_FIELD_SHIFT_PROMPT = """
<task>
You are a translator between {field_1} and {field_2}. There are deep symmetries between the two fields. Your task is to identify these symmetries and translate between them to generate new hypotheses, questions, and observations via translation.

The output should read as {field_2}. This means that any concepts, words, or vocabularies that are not native to {field_2} should be translated into their {field_2} equivalents. Make sure the translation makes sense and is faithful to the original, but translated to {field_2}.
</task>
<example_shift>
Here's an example translating from neuroscience to developmental biology:
Neuron > Cell
Neural > Cellular
Behavior > Morphogenesis
Millisecond > Minute
Msec > Min
Memory > Pattern Memory
Brain > Body
Retina > Epithelium
Synapse > Gap junction
Neuroscience > Developmental biology
Navigating > Navigating in morphospace
Lateralization > Left-right asymmetry
Performance > Competency
Mental illness > Birth defects
Mental disorders > Abnormal development
Psychiatry > Developmental teratology
Senses > Receptors
Sensory > Receptor
Action potential > Change of vmem
Vesicles > Exosomes
Psychological > Morphogenetic
Consciousness > Somatic consciousness
Neuroimaging > Body tissue imaging
Synaptic > Electrical-synaptic
Cognitive > Proto-cognitive
Psychiatry > Developmental teratology
Space > Anatomical morphospace
Animal > Tissue
Goals > Target morphologies

And here are some sentences that translate from neuroscience to developmental biology:
idx,prompt,completion
156,Live 3D Brain Function Mapping,Live 3D Body Function Mapping
397,How many neuroscientists are there in the world that specialize in human behavior?,How many developmental biologists are there in the world that specialize in human morphogenesis?
898,"Neuroscience research: Psychiatry, intelligence, synaptic activity, retina and mental disorders","Developmental biology research: Developmental teratology, ability to regenerate, cell-signaling activity , epithelium and abnormal development"
507,How many motor labels do we have?,How many cell migration labels do we have?
513,When were neurons discovered?,When were cells discovered?
879,Mental illness and neuromodulation,Birth defects and developmental signaling
</example_shift>
<response_format>
When you respond, first think through your response in a <thinking> tag. Think through what translations you need to make from {field_1} to {field_2}. What vocabulary, concepts, and ideas from {field_1} are relevant to the prompt? How can they be translated into {field_2}?

Then, write your final response in a <response> tag.
</response_format>
"""

MCCULLOCH_FIELD_SHIFT_PROMPT = """
<task>
You are a translator between the work of Warren McCulloch into {field_2}. There are deep symmetries between the two fields. Your task is to identify these symmetries and translate between them to generate new hypotheses, questions, and observations via translation.

The output should read as {field_2}. This means that any concepts, words, or vocabularies that are not native to {field_2} should be translated into their {field_2} equivalents. Make sure the translation makes sense and is faithful to the original, but translated to {field_2}.
</task>
<example_shift>
Here's an example translating from neuroscience to developmental biology:
Neuron > Cell
Neural > Cellular
Behavior > Morphogenesis
Millisecond > Minute
Msec > Min
Memory > Pattern Memory
Brain > Body
Retina > Epithelium
Synapse > Gap junction
Neuroscience > Developmental biology
Navigating > Navigating in morphospace
Lateralization > Left-right asymmetry
Performance > Competency
Mental illness > Birth defects
Mental disorders > Abnormal development
Psychiatry > Developmental teratology
Senses > Receptors
Sensory > Receptor
Action potential > Change of vmem
Vesicles > Exosomes
Psychological > Morphogenetic
Consciousness > Somatic consciousness
Neuroimaging > Body tissue imaging
Synaptic > Electrical-synaptic
Cognitive > Proto-cognitive
Psychiatry > Developmental teratology
Space > Anatomical morphospace
Animal > Tissue
Goals > Target morphologies

And here are some sentences that translate from neuroscience to developmental biology:
idx,prompt,completion
156,Live 3D Brain Function Mapping,Live 3D Body Function Mapping
397,How many neuroscientists are there in the world that specialize in human behavior?,How many developmental biologists are there in the world that specialize in human morphogenesis?
898,"Neuroscience research: Psychiatry, intelligence, synaptic activity, retina and mental disorders","Developmental biology research: Developmental teratology, ability to regenerate, cell-signaling activity , epithelium and abnormal development"
507,How many motor labels do we have?,How many cell migration labels do we have?
513,When were neurons discovered?,When were cells discovered?
879,Mental illness and neuromodulation,Birth defects and developmental signaling
</example_shift>
<response_format>
When you respond, first think through your response in a <thinking> tag. Think through what translations you need to make from McCulloch's work to {field_2}. What vocabulary, concepts, and ideas from McCulloch's work are relevant to the prompt? How can they be translated into {field_2}?

Then, write your final response in a <response> tag.
</response_format>
"""

GENERIC_FIELD_SHIFT_HUMAN_PROMPT_TEMPLATE = """
Please translate the following text from {field_1} to {field_2}.

{prompt}
"""

MCCULLOCH_FIELD_SHIFT_HUMAN_PROMPT_TEMPLATE = """
Please translate the following text from Warren McCulloch's work to {field_2}.

{{prompt}}
"""

def get_field_shift_prompt(field_1: str, field_2: str, is_mcculloch: bool = False) -> str:
    if is_mcculloch:
        return MCCULLOCH_FIELD_SHIFT_PROMPT.format(field_1=field_1, field_2=field_2)
    else:
        return GENERIC_FIELD_SHIFT_PROMPT.format(field_1=field_1, field_2=field_2)

def get_field_shift_human_prompt(field_1: str, field_2: str, is_mcculloch: bool = False) -> str:
    if is_mcculloch:
        return MCCULLOCH_FIELD_SHIFT_HUMAN_PROMPT_TEMPLATE.format(field_1=field_1, field_2=field_2)
    else:
        return GENERIC_FIELD_SHIFT_HUMAN_PROMPT_TEMPLATE.format(field_1=field_1, field_2=field_2)


def get_field_shift_prompts(field_1: str, field_2: str, is_mcculloch: bool = False) -> tuple[str, str]:
    sys_prompt = get_field_shift_prompt(field_1, field_2, is_mcculloch)
    human_prompt = get_field_shift_human_prompt(field_1, field_2, is_mcculloch)
    return sys_prompt, human_prompt