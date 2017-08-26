function fin_weights = perceptron(x, w, y, threshold)
  for i = 1:size(x, 1)
    ao = x(i, :)*w;
    ao = ao > threshold;
    if ao != y(i)
      if ao == 0
        w += x(i, :)';
      elseif ao == 1
        w -= x(i, :)';
      endif
    endif
  endfor
  fin_weights = w;
endfunction