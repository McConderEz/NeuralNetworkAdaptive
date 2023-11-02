using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkAdaptive
{
    public class Neuron
    {
        /// <summary>
        /// Веса для перемножения на входящие сигналы
        /// </summary>
        public List<double> Weights { get; }
        /// <summary>
        /// Входящие сигналы
        /// </summary>
        public List<double> Inputs { get; }
        /// <summary>
        /// Тип нейрона(Входной, скрытый, выходной)
        /// </summary>
        public NeuronType NeuronType { get; }
        /// <summary>
        /// Выходное значение после функции активации 
        /// </summary>
        public double Output { get; private set; }


        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            if(inputCount <= 0)
            {
                throw new ArgumentException("Количество входных сигналов не можеть быть меньше или равно 0!",nameof(inputCount));
            }

            NeuronType = type;
            Inputs = new List<double>();
            Weights = new List<double>();

            for(var i = 0;i < inputCount;i++)
            {
                Weights.Add(i);
            }
        }

        public void SetWeights(params double[] weights)
        {
            for(int i = 0;i < weights.Length; i++)
            {
                Weights[i] = weights[i];
            }
        }

        /// <summary>
        /// Инициализация весов
        /// </summary>
        /// <param name="inputCount"></param>
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

        /// <summary>
        /// Перемножение весов на сигналы
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double FeedForward(List<double> inputs)
        {
            //for(int i = 0;i < inputs.Count;i++)
            //{
            //    Inputs[i] = inputs[i];
            //}

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

        /// <summary>
        /// Функция активации Сигмоида
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Pow(Math.E, -x));
        
    }
}
