
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
 [['zero', 'arity0', 'plus'], ['arity0']],

 [['arity0', 'zero', 'mult'], ['zero']],
 [['arity0', 'zero', 'plus'], ['arity0']],
 [['arity0', 'zero', 'minus'], ['arity0']],
 [['arity0', 'zero', 'div'], ['infinite']],

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
 [['scalar', 'variable', 'plus', 'scalar', 'plus'], ['scalar','variable', 'plus']],
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

 [['scalar', 'variable', 'minus', 'scalar', 'minus'], ['scalar', 'variable', 'minus']],
 [['scalar', 'variable', 'plus', 'scalar', 'minus'], ['scalar', 'variable', 'plus']],
 [['scalar', 'variable', 'plus', 'scalar', 'plus'], ['scalar', 'variable', 'plus']],
 [['scalar', 'variable', 'minus', 'scalar', 'plus'], ['scalar', 'variable', 'minus']],

 [['scalar', 'scalar', 'variable', 'minus', 'minus'], ['scalar', 'variable', 'plus']],
 [['scalar', 'scalar', 'variable', 'plus', 'plus'], ['scalar', 'variable', 'plus']],
 [['scalar', 'scalar', 'variable', 'plus', 'minus'], ['scalar', 'variable', 'minus']],
 [['scalar', 'scalar', 'variable', 'minus', 'plus'], ['scalar', 'variable', 'minus']],

 [['scalar', 'variable', 'mult', 'scalar', 'mult'], ['scalar', 'variable', 'mult']],
 [['scalar', 'variable', 'scalar', 'mult', 'mult'], ['scalar', 'variable', 'mult']],
 [['scalar', 'scalar', 'variable', 'mult', 'mult'], ['scalar', 'variable', 'mult']],
 [['scalar', 'scalar', 'variable', 'mult', 'mult'], ['scalar', 'variable', 'mult']],

 [['scalar', 'variable', 'div', 'scalar', 'mult'], ['scalar', 'variable', 'div']],
 [['scalar', 'variable', 'scalar', 'div', 'mult'], ['scalar', 'variable', 'mult']],
 [['scalar', 'scalar', 'variable', 'div', 'mult'], ['scalar', 'variable', 'div']],

 [['scalar', 'variable', 'mult', 'scalar', 'div'], ['scalar', 'variable', 'mult']],
 [['scalar', 'variable', 'scalar', 'mult', 'div'], ['scalar', 'variable', 'div']],
 [['scalar', 'scalar', 'variable', 'mult', 'div'], ['scalar', 'variable', 'div']],

 [['scalar', 'variable', 'div', 'scalar', 'div'], ['scalar', 'variable', 'div']],
 [['scalar', 'variable', 'scalar', 'div', 'div'], ['scalar', 'variable', 'div']],
 [['scalar', 'scalar', 'variable', 'div', 'div'], ['scalar', 'variable', 'mult']],

 [['variable', 'scalar', 'plus', 'scalar', 'plus'], ['scalar', 'variable', 'plus']],
 [['variable', 'scalar', 'minus', 'scalar', 'plus'], ['scalar', 'variable', 'plus']],
 [['variable', 'scalar', 'plus', 'scalar', 'mult'], ['scalar', 'variable', 'mult', 'scalar', 'plus']],
 [['variable', 'scalar', 'minus', 'scalar', 'mult'], ['scalar', 'variable', 'mult', 'scalar', 'plus']],

 [['variable', 'scalar', 'plus', 'scalar', 'minus'], ['scalar', 'variable', 'plus']],
 [['variable', 'scalar', 'minus', 'scalar', 'minus'], ['scalar', 'variable', 'minus']],
 [['variable', 'scalar', 'mult', 'scalar', 'mult'], ['scalar', 'variable', 'mult']],
 [['variable', 'scalar', 'mult', 'scalar', 'div'], ['scalar', 'variable', 'mult']],
 [['variable', 'scalar', 'div', 'scalar', 'mult'], ['scalar', 'variable', 'mult']],
 [['variable', 'scalar', 'div', 'scalar', 'div'], ['scalar', 'variable', 'mult']],

 # (A+(A*(x0))*A
 [['scalar', 'variable', 'mult', 'scalar', 'plus', 'scalar', 'mult'], ['scalar', 'variable', 'mult', 'scalar', 'plus']],
 [['scalar', 'variable', 'mult', 'scalar', 'minus', 'scalar', 'mult'], ['scalar', 'variable', 'mult', 'scalar', 'plus']],
 [['scalar', 'variable', 'mult', 'scalar', 'plus', 'scalar', 'div'], ['scalar', 'variable', 'mult', 'scalar', 'plus']],
 # A*(A-(A/(x0))
  [['scalar', 'scalar', 'variable', 'div', 'minus', 'scalar', 'mult'], ['scalar', 'scalar', 'variable', 'div', 'minus']],
 # ((x0)/A)-A)*A
  [['variable', 'scalar', 'div', 'scalar', 'minus', 'scalar', 'mult'], ['variable', 'scalar', 'div', 'scalar', 'minus']],
  [['variable', 'scalar', 'div', 'scalar', 'plus', 'scalar', 'mult'], ['variable', 'scalar', 'div', 'scalar', 'plus']],
  [['variable', 'scalar', 'mult', 'scalar', 'minus', 'scalar', 'mult'], ['variable', 'scalar', 'mult', 'scalar', 'minus']],
  [['variable', 'scalar', 'mult', 'scalar', 'plus', 'scalar', 'mult'], ['variable', 'scalar', 'mult', 'scalar', 'plus']],

  # A*((x0)**A))*A
  [['scalar', 'variable', 'scalar', 'power', 'mult', 'scalar', 'mult'], ['scalar', 'variable', 'scalar', 'power', 'mult']],

  [['variable', 'fonction', 'variable', 'fonction', 'div'], ['neutral']],
  [['variable', 'fonction', 'variable', 'fonction', 'minus'], ['zero']],
  [['zero', 'power'], ['empty']],
  [['variable', 'mult', 'variable', 'div'], ['empty']],
  [['variable', 'div', 'variable', 'div'], ['empty']],

  [['scalar', 'variable', 'fonction', 'plus'], ['variable', 'fonction', 'scalar', 'plus']],
  [['scalar', 'variable', 'fonction', 'minus'], ['variable', 'fonction', 'scalar', 'minus']]

 ]




mysimplificationrules_no_A = \
 [
 [['1', 'mult'],['empty']],
 [['1', 'power'], ['empty']],
 [['zero', '1', 'plus'], ['1']],
 [['zero', '2', 'plus'], ['2']],

 [['1', 'zero', 'plus'], ['1']],
 [['2', 'zero', 'plus'], ['2']],
 [['1', 'zero', 'minus'], ['1']],
 [['2', 'zero', 'minus'], ['2']],

 [['1', '1', 'plus'], ['2']],
 [['1', '1', 'mult'], ['1']],
 [['1', '1', 'div'], ['1']],
 [['1', '1', 'minus'], ['zero']],
 [['zero', 'sin'], ['zero']],
 [['zero', 'cos'], ['neutral']],
 [['zero', 'exp'], ['neutral']],
 [['1', 'log'], ['zero']],

 [['1', '2', 'mult'], ['2']],
 [['2', '1', 'minus'], ['1']],
 [['2', '1', 'mult'], ['2']],
 [['2', '1', 'div'], ['2']],
 [['1', '2', 'minus'], ['zero', '1', 'minus']],
 [['1', '2', 'power'], ['1']],
 [['2', '2', 'minus'], ['zero']],
 [['2', '2', 'div'], ['1']],

  #size 2
 [['zero', 'plus'], ['empty']],
 [['zero', 'minus'], ['empty']],
 [['zero', 'div'], ['infinite', 'mult']],
 [['neutral', 'mult'], ['empty']],
 [['neutral', 'div'], ['empty']],
 [['neutral', 'power'], ['empty']],
 [['neutral', 'log'], ['zero']],
 [['infinite', 'fonction'], ['infinite']],
 [['log', 'exp'], ['empty']],
 [['exp', 'log'], ['empty']],

 #size three
 [['neutral', '1', 'mult'], ['1']],
 [['neutral', '1', 'plus'], ['2']],
 [['neutral', '1', 'minus'], ['zero']],
 [['neutral', '1', 'div'], ['1']],

 [['neutral', '2', 'mult'], ['2']],
 [['2', 'neutral', 'mult'], ['2']],
 [['2', 'neutral', 'div'], ['2']],
 [['2', 'neutral', 'minus'], ['1']],

 [['neutral', 'variable', 'mult'], ['variable']],
 [['variable', 'neutral', 'mult'], ['variable']],

 [['variable', 'fonction', 'variable', 'fonction', 'div'], ['neutral']],
 [['variable', 'fonction', 'variable', 'fonction', 'minus'], ['zero']],
 [['zero', 'power'], ['empty']],
 [['variable', 'mult', 'variable', 'div'], ['empty']],
 [['variable', 'div', 'variable', 'div'], ['empty']],

# -----1
[['zero', 'arity0', 'mult'], ['zero']],
 [['zero', 'arity0', 'plus'], ['arity0']],
 [['zero', 'arity0', 'div'], ['zero']],
 [['zero', 'arity0', 'plus'], ['arity0']],

 [['arity0', 'zero', 'mult'], ['zero']],
 [['arity0', 'zero', 'plus'], ['arity0']],
 [['arity0', 'zero', 'minus'], ['arity0']],
 [['arity0', 'zero', 'div'], ['infinite']],

 [['variable', 'variable', 'minus'], ['zero']],
 [['variable', 'variable', 'div'], ['neutral']],


#---------
 [['variable', '1', 'mult'], ['variable']],
 [['1', 'variable', 'mult'], ['variable']],
 [['fonction', '1', 'mult'], ['fonction']],
  [['1', 'fonction', 'mult'], ['fonction']],

  [['1', 'log'], ['zero']],
  # size four
  [['variable', 'exp', '2', 'power'], ['variable', '2', 'mult', 'exp']],  # to avoid powers

  [['variable', '1', 'div'], ['variable']],

  [['1', 'variable', 'plus', '1', 'plus'], ['variable', '2', 'plus']],
  [['1', 'variable', 'minus', '1', 'plus'], ['2', 'variable', 'minus']],
  [['1', 'variable', 'plus', '1', 'minus'], ['variable']],
  [['1', 'variable', 'plus', '2', 'minus'], ['variable', '1', 'minus']],
  [['1', 'variable', 'minus', '1', 'minus'], ['zero', 'variable', 'minus']],
  [['2', 'variable', 'plus', '1', 'minus'], ['variable', '1', 'plus']],
  [['2', 'variable', 'minus', '1', 'minus'], ['1', 'variable', 'minus']],
  [['2', 'variable', 'minus', '2', 'minus'], ['zero', 'variable', 'minus']],
  [['2', 'variable', 'plus', '2', 'minus'], ['variable']],

  [['1', 'variable', 'mult'], ['variable']],

  [['variable', 'fonction', 'variable', 'fonction', 'div'], ['neutral']],
  [['variable', 'fonction', 'variable', 'fonction', 'minus'], ['zero']]
 ]
