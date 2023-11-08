using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;

namespace NeuralNetworkAdaptive
{
    public class NeuralNetwork
    {
        /// <summary>
        /// Описание нейронной сети
        /// </summary>
        public Topology Topology { get; }
        /// <summary>
        /// Список слоёв, которые характеризуют нейронную сеть
        /// </summary>
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();
            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }
        
        /// <summary>
        /// Прогон сигналов по нейронной сети 
        /// </summary>
        /// <param name="inputSignals"></param>
        /// <returns></returns>
        public Neuron FeedForward(params double[] inputsSignals)
        {
            if(Topology.InputCount != inputsSignals.Length)
            {
                throw new ArgumentException("Количество входных нейронов не соответствует количеству описанному в топологии!",nameof(inputsSignals.Length));
            }

            SendSignalsToInputNeurons(inputsSignals);
            FeedForwardAllLayersAfterInput();

            if(Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n=>n.Output).First();
            }

        }

        /// <summary>
        /// Метод обучения нейронной сети
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="epoch"></param>
        /// <returns></returns>
        public double Learn(List<Tuple<double, double[]>> dataset, int epoch)
        {
            var error = 0.0;

            for(int i = 0; i < epoch; i++)
            {
                foreach(var data in dataset)
                {
                    error += BackPropagation(data.Item1, data.Item2);
                }
            }
            var result = error / epoch;
            return result;
        }

        /// <summary>
        /// Метод обратного распространения ошибки для балансирования весов
        /// </summary>
        /// <param name="expected"></param>
        /// <param name="inputs"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        private double BackPropagation(double expected, params double[] inputs)
        {

            if(inputs.Length != Topology.InputCount)
            {
                throw new ArgumentException("Количество входных нейронов не соответствует количеству описанному в топологии!", nameof(inputs.Length));
            }

            var actual = FeedForward(inputs).Output;

            var difference = actual - expected;

            //Корректировка весов выходного слоя
            foreach(var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            //Корректировка всех остальных слоёв
            for(int j = Layers.Count - 2; j >= 0; j--)//Выбор слоя
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];
                for(int i = 0; i < layer.NeuronCount; i++)//Выбор нейрона на текущем слое
                {
                    var neuron = layer.Neurons[i];

                    for(int k = 0; k < previousLayer.NeuronCount; k++)//Выбор связи с соединёнными нейронами на предыдущем слое
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }

            }

            var result = difference * difference;
            return result;
        }

        /// <summary>
        /// Прогон сигналов по всем слоям после входного
        /// </summary>
        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayersSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayersSignals);
                }

            }
        }

        /// <summary>
        /// Инициализация входных нейронов и прогон к последующим слоям
        /// </summary>
        /// <param name="inputSignals"></param>
        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            for (var i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }


        /// <summary>
        /// Создание входного слоя
        /// </summary>
        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for(int i = 0;i < Topology.InputCount;i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        /// <summary>
        /// Создание скрытых слоёв
        /// </summary>
        private void CreateHiddenLayers()
        {
            for(int i = 0;i < Topology.HiddenLayers.Count; i++)//Слои
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for(int j = 0;j < Topology.HiddenLayers[i]; j++)//Нейроны в слоях
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);

            }
        }

        /// <summary>
        /// Создание выходного слоя
        /// </summary>
        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }
    }
}
