from django.http import JsonResponse
from rest_framework import generics
from rest_framework import status
from django.core.files.uploadedfile import InMemoryUploadedFile
import pandas as pd
from io import StringIO, BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import traceback
import os
from django.conf import settings
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import rmsprop_v2
from joblib import load

import tensorflow as tf
print(tf.__version__)
class Prediccion(generics.GenericAPIView):
    def get(self, request):
        response = {"pred": 0.0, "error1": 0.0}

        return JsonResponse(response, status=status.HTTP_200_OK)

    def post(self, request):
        response = {}
        
        data = procesar_archivo(request.data["datos"])
        cantidadPrediccion = int(request.data["cantidadPrediccion"])

        if data is not None:
            if not isinstance(data, pd.DataFrame):
                return data
            try:

                if not validar_cantidad_meses(data):
                    response[
                        "msgError"
                    ] = "El archivo no tiene el formato correcto. Debe contener 12 meses consecutivos de datos."
                    return JsonResponse(response, status=status.HTTP_400_BAD_REQUEST)
                
                if not validar_cabeceras_correctas(data):
                    response[
                        "msgError"
                    ] = "El archivo no tiene el formato correcto. Debe contener las columnas 'Año', 'Mes' y 'Toneladas'."
                    return JsonResponse(response, status=status.HTTP_400_BAD_REQUEST)

                data = arreglar_fecha_dataFrame(data)

                if not validar_separacion_por_mes(data):
                    response[
                        "msgError"
                    ] = "El archivo no tiene el formato correcto. Debe contener 12 meses consecutivos de datos."
                    return JsonResponse(response, status=status.HTTP_400_BAD_REQUEST)

                if not validar_anio_maximo(data):
                    response[
                        "msgError"
                    ] = "Lo sentimos, el archivo no tiene el formato correcto. El año no puede ser superior a 2024 ni inferior a 2000 para la predicción."
                    return JsonResponse(response, status=status.HTTP_400_BAD_REQUEST)

                resultadoModelo = [
                    1171.2379,
                    1178.2849,
                    1185.3320,
                    1192.3790,
                    1199.4261,
                    1206.4731,
                ]
                
                modelo, escalador = cargar_modelo_y_escalador()
                
                predicciones_nuevas = predecir(data, modelo, escalador)
                
                print("** Predicciones nuevas **")
                print(predicciones_nuevas)
                
                prediccion_fechas = prediccion_con_fechas(data, predicciones_nuevas, cantidadPrediccion)
                
                dataframe_resultado_completo = data.combine_first(prediccion_fechas)
                
                
                print("** Dataframe completo **")
                
                print(dataframe_resultado_completo.dtypes)
                
                grafica = generar_grafica(dataframe_resultado_completo)
                
                
                prediccion_fechas.index = prediccion_fechas.index.strftime('%Y-%m')
                prediccion_fechas_json = prediccion_fechas.to_json(orient='split')
                
                
                response["grafica"] = grafica
                response["prediccion"] = prediccion_fechas_json

                return JsonResponse(response, status=status.HTTP_200_OK)
            except:
                response["msgError"] = "Error al procesar el archivo. Intente nuevamente."
                print(traceback.format_exc())
                return JsonResponse(response, status=status.HTTP_400_BAD_REQUEST)
        else:
            response["msgError"] = "El archivo no es de tipo InMemoryUploadedFile."
            return JsonResponse(response, status=status.HTTP_400_BAD_REQUEST)


def procesar_archivo(archivo):
    if isinstance(archivo, InMemoryUploadedFile):
        contenido = archivo.read().decode("utf-8")

        dataframe = pd.read_csv(StringIO(contenido), sep=";")
        dataframe = dataframe.dropna()
        if not validar_cabeceras_correctas(dataframe):
            response = {}
            response[
                        "msgError"
                    ] = "El archivo no tiene el formato correcto. Debe contener las columnas 'Año', 'Mes' y 'Toneladas'."
            return JsonResponse(response, status=status.HTTP_400_BAD_REQUEST)
        print("dataframe isInstance", isinstance(dataframe, pd.DataFrame))
        dataframe['Tonelada'] = pd.to_numeric(dataframe['Tonelada'].str.replace('.', '').str.replace(',', '.'), errors='coerce')

        return dataframe

    else:
        return None


def validar_cantidad_meses(dataframe, cantidad_mes_valida=12):
    return dataframe.shape[0] == cantidad_mes_valida

def validar_cabeceras_correctas(dataframe):
    return dataframe.columns.tolist() == ["Año", "Mes", "Tonelada"]

def arreglar_fecha_dataFrame(dataframe):
    dataframe["Fecha"] = pd.to_datetime(
        dataframe["Año"].astype(str) + "-" + dataframe["Mes"].astype(str),
        format="%Y-%m",
    )
    dataframe = dataframe.drop(["Año", "Mes"], axis=1)
    dataframe = dataframe.set_index("Fecha")
    dataframe.sort_index(inplace=True)

    return dataframe


def validar_separacion_por_mes(dataframe):
    diferencias = dataframe.index.to_series().diff().dt.days

    # Verificar que todas las diferencias sean aproximadamente de un mes
    tolerancia_dias = 2
    return (
        diferencias.dropna().between(29 - tolerancia_dias, 31 + tolerancia_dias).all()
    )


def validar_anio_maximo(dataframe, anio_maximo=2024, anio_minimo=2000):
    return (
        dataframe.index.year.max() <= anio_maximo
        and dataframe.index.year.min() >= anio_minimo
    )
    
def custom_object_dict():
    return {'optimizador': rmsprop_v2.RMSprop()}

def cargar_modelo_y_escalador():
    # Rutas a tus archivos estáticos
    print(os.path.join(settings.MEDIA_ROOT, 'modelo','modelo.keras'))
    print(os.path.join(settings.MEDIA_ROOT, 'modelo','tescalador.joblib'))
    modelo_path = os.path.join(settings.MEDIA_ROOT, 'modelo','modelo.keras')
    escalador_path = os.path.join(settings.MEDIA_ROOT, 'modelo','escalador.joblib')

    # Cargar el modelo
    modelo = load_model(modelo_path,custom_objects={'RMSprop': rmsprop_v2.RMSprop(learning_rate=1e-5)})

    # Cargar el escalador
    escalador = load(escalador_path)

    return modelo, escalador
    
def predecir(x, model, scaler, input_length = 12):
    
    x_prueba = x['Tonelada'].values[-input_length:].reshape((input_length, 1))
    
    print("** x_prueba **")
    print(x_prueba)
    
    x_prueba_escalada = scaler.transform(x_prueba)
    
    print("\n** x_prueba_escalada **")
    print(x_prueba_escalada)
    x_prueba_escalada = x_prueba_escalada.reshape((1, input_length, 1))
    
    print("\n** x_prueba_escalada reshaped **")
    print(x_prueba_escalada)

    # Calcular predicción escalada en el rango de -1 a 1
    y_pred_s = model.predict(x_prueba_escalada,verbose=0)

    # Llevar la predicción a la escala original
    y_pred = scaler.inverse_transform(y_pred_s)

    return y_pred.flatten()

def prediccion_con_fechas(dataframe, ValoresPrediccion, cantidadPrediccion):
    ultima_fecha = dataframe.index[-1]
    nuevas_fechas = pd.date_range(start=ultima_fecha, periods=cantidadPrediccion + 1, freq='MS')[1:]
    resultadoModelo_con_fechas = pd.DataFrame({'Tonelada':ValoresPrediccion[0:cantidadPrediccion]}, index=nuevas_fechas)
    
    return resultadoModelo_con_fechas

    
    
def generar_grafica(dataframe):
    dataframe['Tonelada'].plot(marker='o', linestyle='-')

    # Añadir etiquetas y título
    plt.xlabel('Fecha')
    plt.ylabel('Tonelada')
    plt.title('Gráfico de Tonelada a lo largo del tiempo')
    # Mostrar la gráfica
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    
    return img_base64
    
    

    