
mysimplificationrules = \
 [
 #size 2
 [['zero', 'plus'], ['empty']],
 [['zero', 'minus'], ['empty']],
 [['zero', 'div'], ['infinite', 'mult']],
 [['neutral', 'mult'], ['empty']],
 [['neutral', 'div'], ['empty']],
 [['neutral', 'power'], ['empty']],
 [['neutral', 'log'], ['zero']],
 [['scalar', 'fonction'], ['scalar']],
 [['infinite', 'fonction'], ['infinite']],
 [['log', 'exp'], ['empty']],
 [['exp', 'log'], ['empty']],

 #size three
 [['neutral', 'scalar', 'mult'], ['scalar']],
 [['neutral', 'scalar', 'plus'], ['scalar']],
 [['neutral', 'scalar', 'minus'], ['scalar']],
 [['neutral', 'scalar', 'div'], ['scalar']],
 [['neutral', 'variable', 'mult'], ['variable']],
 [['scalar', 'neutral', 'mult'], ['scalar']],
 [['scalar', 'neutral', 'plus'], ['scalar']],
 [['scalar', 'neutral', 'minus'], ['scalar']],
 [['scalar', 'neutral', 'div'], ['scalar']],
 [['variable', 'neutral', 'mult'], ['variable']],
 [['zero', 'arity0', 'mult'], ['zero']],
 [['zero', 'arity0', 'plus'], ['arity0']],
 [['zero', 'arity0', 'div'], ['zero']],
 [['arity0', 'zero', 'mult'], ['zero']],
 [['arity0', 'zero', 'plus'], ['arity0']],
 [['arity0', 'zero', 'minus'], ['arity0']],
 [['zero', 'arity0', 'plus'], ['arity0']],
 [['zero', 'arity0', 'div'], ['infinite']],
 [['scalar', 'scalar', 'allops'], ['scalar']],
 [['variable', 'variable', 'minus'], ['zero']],
 [['variable', 'variable', 'div'], ['neutral']],

     #size four
 [['scalar', 'power', 'scalar', 'power'], ['scalar', 'power']],
 [['variable', 'exp', 'scalar', 'power'], ['variable', 'scalar', 'mult', 'exp']],  #to avoid powers
 [['variable', 'scalar', 'mult', 'log'], ['scalar', 'variable', 'log', 'plus']],
 [['scalar', 'variable', 'mult', 'log'], ['scalar', 'variable', 'log', 'plus']],
 [['variable', 'scalar', 'div', 'log'], ['variable', 'log', 'scalar', 'minus']],
 [['scalar', 'variable', 'div', 'log'], ['scalar', 'variable', 'log', 'minus']],
 [['scalar', 'plus', 'scalar', 'plus'], ['scalar', 'plus']],
 [['scalar', 'plus', 'scalar', 'minus'], ['scalar', 'plus']],
 [['scalar', 'minus', 'scalar', 'plus'], ['scalar', 'plus']],
 [['scalar', 'minus', 'scalar', 'minus'], ['scalar', 'plus']],
 [['scalar', 'mult', 'scalar', 'div'], ['scalar', 'mult']],
 [['scalar', 'div', 'scalar', 'mult'], ['scalar', 'plus']],
 [['scalar', 'mult', 'scalar', 'mult'], ['scalar', 'mult']],
 [['scalar', 'div', 'scalar', 'div'], ['scalar', 'div']],
 [['scalar','variable', 'plus', 'scalar', 'plus'], ['scalar','variable', 'plus']],
 [['scalar', 'variable', 'minus', 'scalar', 'plus'], ['scalar', 'variable', 'minus']],
 [['scalar', 'variable', 'plus', 'scalar', 'minus'], ['scalar', 'variable', 'plus']],
 [['scalar', 'variable', 'minus', 'scalar', 'minus'], ['scalar', 'variable', 'minus']],
 [['scalar', 'variable', 'mult', 'scalar', 'mult'], ['scalar', 'variable', 'mult']],
 [['scalar', 'variable', 'mult', 'scalar', 'div'], ['scalar', 'variable', 'mult']],
 [['scalar', 'variable', 'div', 'scalar', 'mult'], ['scalar', 'variable', 'div']],
 [['scalar', 'variable', 'div', 'scalar', 'div'], ['scalar', 'variable', 'div']],

 [['variable', 'fonction', 'scalar', 'plus', 'scalar', 'div'], ['variable', 'fonction', 'scalar', 'mult', 'scalar', 'plus']],
 [['variable', 'fonction', 'scalar', 'minus', 'scalar', 'div'], ['variable', 'fonction', 'scalar', 'mult', 'scalar', 'plus']],

 [['variable', 'fonction', 'scalar', 'plus', 'scalar', 'mult'], ['variable', 'fonction', 'scalar', 'mult', 'scalar', 'plus']],
 [['variable', 'fonction', 'scalar', 'minus', 'scalar', 'mult'], ['variable', 'fonction', 'scalar', 'mult', 'scalar', 'plus']],

 [['variable', 'scalar', 'plus', 'scalar', 'plus'], ['scalar', 'variable', 'plus']],
 [['variable', 'scalar', 'minus', 'scalar', 'plus'], ['scalar', 'variable', 'plus']],
 [['variable', 'scalar', 'plus', 'scalar', 'mult'], ['scalar', 'variable', 'mult', 'scalar', 'plus']],
 [['variable', 'scalar', 'minus', 'scalar', 'mult'], ['scalar', 'variable', 'mult', 'scalar', 'plus']],

 [['variable', 'scalar', 'plus', 'scalar', 'minus'], ['scalar', 'variable', 'plus']],
 [['variable', 'scalar', 'minus', 'scalar', 'minus'], ['scalar', 'variable', 'minus']],
 [['variable', 'scalar', 'mult', 'scalar', 'mult'], ['scalar', 'variable', 'mult']],
 [['variable', 'scalar', 'mult', 'scalar', 'div'], ['scalar', 'variable', 'mult']],
 [['variable', 'scalar', 'div', 'scalar', 'mult'], ['scalar', 'variable', 'mult']],
 [['variable', 'scalar', 'div', 'scalar', 'div'], ['scalar', 'variable', 'mult']]

 ]

