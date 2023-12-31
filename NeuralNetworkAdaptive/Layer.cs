﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkAdaptive
{
    public class Layer
    {
        /// <summary>
        /// Нейроны в слое
        /// </summary>
        public List<Neuron> Neurons { get; }
        public int NeuronCount => Neurons?.Count ?? 0;
        /// <summary>
        /// Тип нейрона будет характеризовать и тип слоя
        /// </summary>
        public NeuronType Type { get; }

        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        {
            
            foreach(Neuron neuron in neurons)
            {
                if(neuron.NeuronType != type)
                    throw new ArgumentException("Не соответствие типа нейрона с типом слоя!",nameof(neuron));
            }

            Neurons = neurons;
            Type = type;
        }

        /// <summary>
        /// Получение результирующих сигналов с прошлых слоёв
        /// </summary>
        /// <returns></returns>
        public List<double> GetSignals()
        {
            var result = new List<double>();
            foreach(var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }

            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
