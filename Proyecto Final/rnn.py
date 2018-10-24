import tensorflow as tf

class RNN(object):
    
    def __init__ (self, input, hidden, output):
        self.input_nodes = input
        self.hidden_nodes = hidden
        self.output_nodes = output

       #Create random tensor weighs
        self.weights_ih = tf.random_normal([self.hidden_nodes, self.input_nodes])
        self.weights_ho = tf.random_normal([self.output_nodes, self.hidden_nodes])
        
        #random bias tensor
        self.bias_h = tf.random_normal([self.hidden_nodes, 1])
        self.bias_o = tf.random_normal([self.output_nodes, 1])

    def predict(self, input_array):
      
      # Generating the Hidden Outputs
      inputs = tf.Variable(input_array)
      hidden = tf.multiply(self.weights_ih,inputs)

      
      hidden = tf.add(hidden,self.bias_h)
      # activation function!
      hidden = tf.sigmoid(hidden)
      
      #console.log([inputs,hidden,self.weights_ih,self.weights_ho])
      
  

      # Generating the output's output!
      output = tf.multiply(tf.transpose(self.weights_ho),hidden)
      output = tf.add(output,self.bias_o)
      output = tf.sigmoid(output)
  
      # Sending back to the caller!
      return output
    

brain = RNN(2,2,2)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print(type(brain.predict([0.5,0.5]).eval(sess)))