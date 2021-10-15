using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetworkV2;
using Energy;
using System.Drawing;
using System.Runtime.InteropServices;


namespace NeuralNetworksV2
{
    class Program
    {

        [DllImport("kernel32.dll", EntryPoint = "GetConsoleWindow", SetLastError = true)]
        private static extern IntPtr GetConsoleHandle();

        static void Main(string[] args)
        {
            PixelEnergy pixel = new PixelEnergy();

            string link = "C:\\Users\\сервер\\Desktop\\NeuralNetworksV2\\img\\22.png";
            int barrier = 4;


            Vector[] X = pixel.LowEnergyPixelsCoordinates(link, barrier);
            Vector[] Y = pixel.LowEnergyPixels(link, barrier);

            Network network = new Network(new int[] { 2, 4, 3 });

            double alpha = 0.5;
            double eps = 1e-4;
            int selector = 0;
            int output = 10;
            int x = 100;

            int indextIR = 0; // записей
            int iteration = 0; // вывод при этирации
            double error; // ошибка эпохи
            do
            {
                error = 0; // обнуляем ошибку
                           // проходимся по всем элементам обучающего множества
                for (int i = 0; i < X.Length; i++)
                {
                    network.Forward(X[i], selector); // прямое распространение сигнала
                    network.Backward(Y[i], ref error); // обратное распространение ошибки
                    network.UpdateWeights(alpha); // обновление весовых коэффициентов
                }

                iteration++;

                Console.SetCursorPosition(0, 0);
                Console.WriteLine("error: {0}", error); // выводим в консоль номер эпохи и величину ошибку
                Console.WriteLine((iteration * 100) / output + "%");
                Console.WriteLine(indextIR + " кол. этир.");

                if (iteration == output)
                {
                    network.OutputLayers(); // сохранение нейронов

                    Bitmap bmp = new Bitmap(link);


                    for (int i = 0; i < X.GetLength(0); i++)
                    {
                        Vector OutputValues = network.Forward(X[i], selector);

                        Color gotColor = Color.FromArgb(255, Convert.ToInt32(OutputValues[0] * 255), Convert.ToInt32(OutputValues[1] * 255), Convert.ToInt32(OutputValues[2] * 255));

                        bmp.SetPixel(Convert.ToInt32(X[i][0] * bmp.Width), Convert.ToInt32(X[i][1] * bmp.Height), gotColor);

                    }


                    using (var graphics = Graphics.FromHwnd(GetConsoleHandle()))
                    using (var image = bmp)
                        graphics.DrawImage(image, 50 + x, 50, 250 + x, 200);


                    indextIR++;
                    iteration = 0;
                }

            } while (error > eps);

            Console.ReadKey();

        }
    }
}
