
mysimplificationrules_with_A = \
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
  [['zero', 'zero', 'mult'], ['zero']],
  [['zero', 'zero', 'plus'], ['zero']],
  [['zero', 'zero', 'minus'], ['zero']],

  [['zero', 'scalar', 'power'], ['zero']],
  [['zero', 'variable', 'power'], ['zero']],

  [['neutral', 'scalar', 'mult'], ['scalar']],
 [['neutral', 'scalar', 'plus'], ['scalar']],
 [['neutral', 'scalar', 'minus'], ['scalar']],
 [['neutral', 'scalar', 'div'], ['scalar']],
 [['neutral', 'variable', 'mult'], ['variable']],
 [['scalar', 'neutral', 'plus'], ['scalar']],
 [['scalar', 'neutral', 'minus'], ['scalar']],

 [['zero', 'arity0', 'mult'], ['zero']],
 [['zero', 'arity0', 'plus'], ['arity0']],
 [['zero', 'arity0', 'div'], ['zero']],
 [['arity0', 'zero', 'mult'], ['zero']],

 [['scalar', 'scalar', 'allops'], ['scalar']],

 [['variable', 'variable', 'minus'], ['zero']],
 [['variable', 'variable', 'div'], ['neutral']],

     #size four

 [['variable', 'fonction', 'variable', 'fonction', 'div'], ['neutral']],
 [['variable', 'fonction', 'variable', 'fonction', 'minus'], ['zero']],
 [['variable', 'mult', 'variable', 'div'], ['empty']],
 [['variable', 'div', 'variable', 'div'], ['empty']],

 [['scalar', 'power', 'scalar', 'power'], ['scalar', 'power']],
 [['variable', 'exp', 'scalar', 'power'], ['variable', 'scalar', 'mult', 'exp']],  #to avoid powers
 [['variable', 'scalar', 'mult', 'log'], ['variable', 'log', 'scalar', 'plus']],
 [['scalar', 'variable', 'mult', 'log'], ['variable', 'log', 'scalar', 'plus']],
 [['variable', 'scalar', 'div', 'log'], ['variable', 'log', 'scalar', 'minus']],
 [['scalar', 'variable', 'div', 'log'], ['scalar', 'variable', 'log', 'minus']],

 [['scalar', 'plus', 'scalar', 'plus'], ['scalar', 'plus']],
 [['scalar', 'plus', 'scalar', 'minus'], ['scalar', 'plus']],

 [['scalar', 'minus', 'scalar', 'plus'], ['scalar', 'plus']],
 [['scalar', 'minus', 'scalar', 'minus'], ['scalar', 'plus']],

 [['scalar', 'mult', 'scalar', 'div'], ['scalar', 'mult']],
 [['scalar', 'div', 'scalar', 'mult'], ['scalar', 'mult']],

 [['scalar', 'mult', 'scalar', 'mult'], ['scalar', 'mult']],
 [['scalar', 'div', 'scalar', 'div'], ['scalar', 'mult']],

 #size 5
 [['scalar', 'variable', 'plus', 'scalar', 'plus'], ['scalar','variable', 'plus']],
 [['scalar', 'variable', 'minus', 'scalar', 'plus'], ['scalar', 'variable', 'minus']],
 [['scalar', 'variable', 'plus', 'scalar', 'minus'], ['scalar', 'variable', 'plus']],
 [['scalar', 'variable', 'minus', 'scalar', 'minus'], ['scalar', 'variable', 'minus']],
 [['scalar', 'variable', 'mult', 'scalar', 'mult'], ['scalar', 'variable', 'mult']],
 [['scalar', 'variable', 'mult', 'scalar', 'div'], ['scalar', 'variable', 'mult']],
 [['scalar', 'variable', 'div', 'scalar', 'mult'], ['scalar', 'variable', 'div']],
 [['scalar', 'variable', 'div', 'scalar', 'div'], ['scalar', 'variable', 'div']],

 [['scalar', 'scalar', 'variable', 'plus', 'plus'], ['scalar', 'variable', 'plus']],
 [['scalar', 'scalar', 'variable', 'plus', 'minus'], ['scalar', 'variable', 'minus']],
 [['scalar', 'scalar', 'variable', 'minus', 'plus'], ['scalar', 'variable', 'minus']],
 [['scalar', 'scalar', 'variable', 'minus', 'minus'], ['scalar', 'variable', 'plus']],

 [['scalar', 'scalar', 'variable', 'mult', 'mult'], ['scalar', 'variable', 'mult']],
 [['scalar', 'scalar', 'variable', 'div', 'mult'], ['scalar', 'variable', 'div']],
 [['scalar', 'scalar', 'variable', 'mult', 'div'], ['scalar', 'variable', 'div']],
 [['scalar', 'scalar', 'variable', 'div', 'div'], ['scalar', 'variable', 'mult']],


 [['scalar', 'variable', 'scalar', 'mult', 'mult'], ['scalar', 'variable', 'mult']],
 [['scalar', 'variable', 'scalar', 'div', 'mult'], ['scalar', 'variable', 'mult']],
 [['scalar', 'variable', 'scalar', 'mult', 'div'], ['scalar', 'variable', 'div']],
 [['scalar', 'variable', 'scalar', 'div', 'div'], ['scalar', 'variable', 'div']],

 [['scalar', 'variable', 'scalar', 'plus', 'plus'], ['scalar', 'variable', 'plus']],
 [['scalar', 'variable', 'scalar', 'plus', 'minus'], ['scalar', 'variable', 'minus']],
 [['scalar', 'variable', 'scalar', 'minus', 'plus'], ['scalar', 'variable', 'plus']],
 [['scalar', 'variable', 'scalar', 'minus', 'minus'], ['scalar', 'variable', 'minus']],

  # size 6
  [['variable', 'fonction', 'scalar', 'plus', 'scalar', 'div'], ['variable', 'fonction', 'scalar', 'mult', 'scalar','plus']],
  [['variable', 'fonction', 'scalar', 'minus', 'scalar', 'div'], ['variable', 'fonction', 'scalar', 'mult', 'scalar', 'plus']],
  [['variable', 'fonction', 'scalar', 'plus', 'scalar', 'mult'], ['variable', 'fonction', 'scalar', 'mult', 'scalar', 'plus']],
  [['variable', 'fonction', 'scalar', 'minus', 'scalar', 'mult'], ['variable', 'fonction', 'scalar', 'mult', 'scalar', 'plus']],

 #size 7

  # type (A+(A*(x0))*A
  [['scalar', 'plus', 'scalar', 'mult'], ['scalar', 'mult', 'scalar', 'plus']],
  [['scalar', 'plus', 'scalar', 'div'], ['scalar', 'mult', 'scalar', 'plus']],
  [['scalar', 'minus', 'scalar', 'mult'], ['scalar', 'mult', 'scalar', 'minus']],
  [['scalar', 'minus', 'scalar', 'div'], ['scalar', 'div', 'scalar', 'minus']],

  [['scalar', 'variable', 'mult', 'scalar', 'minus', 'scalar', 'mult'], ['scalar', 'variable', 'mult', 'scalar', 'plus']],
  # A*(A-(A/(x0))
  [['scalar', 'scalar', 'variable', 'div', 'minus', 'scalar', 'mult'], ['scalar', 'scalar', 'variable', 'div', 'minus']],
  # A*((x0)**A))*A
  [['scalar', 'variable', 'scalar', 'power', 'mult', 'scalar', 'mult'], ['scalar', 'variable', 'scalar', 'power', 'mult']],

 ]




mysimplificationrules_no_A = \
 [
 [['one'], ['neutral']],


 [['neutral', 'mult'],['empty']],
 [['neutral', 'power'], ['empty']],
 [['zero', 'sin'], ['zero']],
 [['zero', 'cos'], ['neutral']],
 [['zero', 'exp'], ['neutral']],
 [['neutral', 'log'], ['zero']],
 [['zero', 'plus'], ['empty']],
 [['zero', 'minus'], ['empty']],
 [['zero', 'div'], ['infinite', 'mult']],
 [['neutral', 'div'], ['empty']],
 [['log', 'exp'], ['empty']],
 [['exp', 'log'], ['empty']],

  # size three
  [['zero', 'zero', 'mult'], ['zero']],
  [['zero', 'zero', 'plus'], ['zero']],
  [['zero', 'zero', 'minus'], ['zero']],

  [['zero', 'two', 'power'], ['zero']],

  [['zero', 'arity0', 'mult'], ['zero']],
 [['zero', 'arity0', 'plus'], ['arity0']],
 [['zero', 'arity0', 'div'], ['zero']],
 [['arity0', 'zero', 'mult'], ['zero']],
 [['neutral', 'neutral', 'plus'], ['two']],
 [['neutral', 'neutral', 'minus'], ['zero']],
  [['neutral', 'neutral', 'div'], ['neutral']],

  [['neutral', 'two', 'mult'], ['two']],
 [['two', 'neutral', 'minus'], ['neutral']],
 [['neutral', 'two', 'minus'], ['zero', 'neutral', 'minus']],
 [['neutral', 'two', 'power'], ['neutral']],
 [['two', 'two', 'minus'], ['zero']],
 [['two', 'two', 'div'], ['neutral']],
 [['variable', 'variable', 'minus'], ['zero']],
 [['variable', 'variable', 'div'], ['neutral']],
 [['neutral', 'variable', 'mult'], ['variable']],

 #size four

 [['variable', 'mult', 'variable', 'div'], ['empty']],
 [['variable', 'div', 'variable', 'div'], ['empty']],
 #[['variable', 'exp', 'two', 'power'], ['variable', 'two', 'mult', 'exp']],  # to avoid powers

  # size 5

 [['variable', 'fonction', 'variable', 'fonction', 'div'], ['neutral']],
 [['variable', 'fonction', 'variable', 'fonction', 'minus'], ['zero']],
 [['neutral', 'variable', 'plus', 'neutral', 'plus'], ['variable', 'two', 'plus']],
 [['neutral', 'variable', 'minus', 'neutral', 'plus'], ['two', 'variable', 'minus']],
 [['neutral', 'variable', 'plus', 'neutral', 'minus'], ['variable']],
 [['neutral', 'variable', 'plus', 'two', 'minus'], ['variable', 'neutral', 'minus']],
 [['neutral', 'variable', 'minus', 'neutral', 'minus'], ['zero', 'variable', 'minus']],
 [['two', 'variable', 'plus', 'neutral', 'minus'], ['variable', 'neutral', 'plus']],
 [['two', 'variable', 'minus', 'neutral', 'minus'], ['neutral', 'variable', 'minus']],
 [['two', 'variable', 'minus', 'two', 'minus'], ['zero', 'variable', 'minus']],
 [['two', 'variable', 'plus', 'two', 'minus'], ['variable']],

 [['variable', 'fonction', 'variable', 'fonction', 'div'], ['neutral']],
 [['variable', 'fonction', 'variable', 'fonction', 'minus'], ['zero']]

 ]
