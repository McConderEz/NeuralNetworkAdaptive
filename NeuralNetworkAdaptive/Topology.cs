using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkAdaptive
{
    public class Topology
    {
        public int InputCount { get; }
        public int OutputCount { get; }
        public double LearningRate { get; }

        /// <summary>
        /// Список скрытых слоёв
        /// </summary>
        public List<int> HiddenLayers { get; }
        /// <summary>
        /// Конструктор создания описания нейронной сети
        /// </summary>
        /// <param name="inputCount"> количество входных сигналов/параметров(входные нейроны)</param>
        /// <param name="outputCount"> количество выходных сигналов(результирующих)</param>
        /// <param name="layers">массив, который характеризует количество скрытых слоёв и количество нейронов в них</param>
        public Topology(int inputCount, int outputCount, double learningRate ,params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;
            HiddenLayers = new List<int>();
            HiddenLayers.AddRange(layers);
        }

    }
}
