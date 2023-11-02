using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        public Neuron FeedForward(List<double> inputSignals)
        {
            if(Topology.InputCount != inputSignals.Count)
            {
                throw new ArgumentException("Количество входны нейронов не соответствует количеству описанному в топологии!",nameof(inputSignals.Count));
            }

            SendSignalsToInputNeurons(inputSignals);
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
        private void SendSignalsToInputNeurons(List<double> inputSignals)
        {
            for (var i = 0; i < inputSignals.Count; i++)
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
