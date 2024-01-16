from django.http import JsonResponse
from rest_framework import generics
from rest_framework import status
from django.shortcuts import render
import traceback

class Prediccion(generics.GenericAPIView):
    def get(self, request):
        response = {"pred": 0.0, "error": 0.0}

        return JsonResponse(response, status=status.HTTP_200_OK)
