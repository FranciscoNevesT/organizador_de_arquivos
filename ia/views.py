from django.shortcuts import render, redirect
from django.template import loader
from django.http import HttpResponse

# Create your views here.
def train_model(request):
  if request.method == 'POST':
    # Get user input (e.g., training parameters)
    parameters = request.POST.get('parameters')
    # Simulate training with the input (replace with actual training logic)
    print(f"Training model with parameters: {parameters}")
    return redirect('/ia')  # Redirect to home page after training
  else:
    return render(request, 'ia/train_form.html')