using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkAdaptive
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }


        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Inputs = new List<double>();
            Weights = new List<double>();

        }


        private void InitWeightsRandomValue(int inputCount)
        {
            var rnd = new Random();

            for(int i = 0;i < inputCount; i++)
            {
                if(NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());
                }
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            for(int i = 0;i < inputs.Count;i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for(int i = 0;i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }
            if(NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }

            return Output;
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Pow(Math.E, -x));
        
    }
}
