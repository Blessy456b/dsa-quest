from supabase import create_client
url = "https://wjgmdtqgvbloxxxxinid.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqZ21kdHFndmJsb3h4eHhpbmlkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1NTU5NDksImV4cCI6MjA3OTEzMTk0OX0.COBk91bbC94y6M_RpVJpAVoMPdQEt6g3Pvxs34hBc8g"
supabase = create_client(url, key)
print(supabase.table("users_progress").select("*").limit(1).execute())
