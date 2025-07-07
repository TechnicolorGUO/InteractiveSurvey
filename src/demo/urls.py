from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('delete_files/', views.delete_files, name='delete_files'),
    path('generate_arxiv_query/', views.generate_arxiv_query, name='generate_arxiv_query'),
    path('get_survey_id/', views.get_survey_id, name='get_survey_id'),
    path('get_surveys/', views.get_surveys, name='get_surveys'),
    path('generate_pdf/', views.generate_pdf, name='generate_pdf'),
    path('generate_pdf_from_tex/', views.generate_pdf_from_tex, name='generate_pdf_from_tex'),
    path('save_outline/', views.save_outline, name='save_outline'),
    path("download_pdfs/", views.download_pdfs, name="download_pdfs"),
    path('save_updated_cluster_info', views.save_updated_cluster_info, name='save_updated_cluster_info'),
    path('get_operation_progress/', views.get_operation_progress, name='get_operation_progress'),
    # path('test_async_simple/', views.test_async_simple, name='test_async_simple'),
    url(r'^get_topic$', views.get_topic, name='get_topic'),
    url(r'^get_survey$', views.get_survey, name='get_survey'),
    url(r'^automatic_taxonomy$', views.automatic_taxonomy, name='automatic_taxonomy'),
    url(r'^upload_refs$', views.upload_refs, name='upload_refs'),
    url(r'^annotate_categories$', views.annotate_categories, name='annotate_categories'),
    url(r'^select_sections$', views.select_sections, name='select_sections'),
    
]
