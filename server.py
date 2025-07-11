import json
import sys
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from datetime import datetime, timezone, timedelta
from skyfield.api import load, wgs84
from math import sin, cos, tan, atan, atan2, radians, degrees
import numpy as np
import os
from pathlib import Path
from functools import lru_cache
import requests
import urllib.request
import ssl
import csv

try:
    import swisseph as swe
    SWISSEPH_AVAILABLE = True
    print("Swiss Ephemeris disponible para cálculos de casas precisos")
except ImportError:
    SWISSEPH_AVAILABLE = False
    print("NOTA: Swiss Ephemeris no disponible - usando cálculos aproximados")
    print("Para máxima precisión, instala pyswisseph (requiere Visual C++ Build Tools)")

app = Flask(__name__)
# Configurar CORS correctamente
CORS(app, resources={r"/*": {"origins": "*"}})

# Variables globales para recursos precargados
eph = None
ts = None
time_zone_df = None

API_KEY = "e19afa2a9d6643ea9550aab89eefce0b"

# Precarga de recursos al inicio
def preload_resources():
    global eph, ts, time_zone_df
    
    print("Precargando recursos...")
    
    # Cargar efemérides desde GitHub
    try:
        # Cargar desde archivo local
        eph_path = Path('de421.bsp')
        if not eph_path.exists():
            # Intentar cargar desde la carpeta docs
            eph_path = Path('docs') / 'de421.bsp'
        
        print(f"Cargando efemérides desde: {eph_path}")
        eph = load(str(eph_path))
    except Exception as e:
        print(f"Error cargando efemérides: {e}")
        # Intento alternativo
        try:
            print("Intentando cargar efemérides alternativas...")
            eph = load('de440s.bsp')
        except Exception as e2:
            print(f"Error en carga alternativa: {e2}")
            sys.exit(1)  # Salir si no se pueden cargar las efemérides
    
    ts = load.timescale()
    
    # Cargar zona horaria desde CSV
    try:
        time_zone_df = []
        with open('time_zone.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) >= 6:  # asegurarse de que hay suficientes columnas
                    time_zone_df.append({
                        'timezone': row[0],
                        'country_code': row[1],
                        'abbreviation': row[2],
                        'timestamp': int(row[3]) if row[3].isdigit() else 0,
                        'utc_offset': float(row[4]) if row[4].replace('.', '', 1).isdigit() else 0,
                        'dst': int(row[5]) if row[5].isdigit() else 0
                    })
        print(f"Cargado archivo de zonas horarias: {len(time_zone_df)} entradas")
    except Exception as e:
        print(f"Error cargando zonas horarias: {e}")
        time_zone_df = []
    
    print("Recursos precargados correctamente")

# Cachear obtención de datos de ciudad con lru_cache nativo
@lru_cache(maxsize=100)
def obtener_datos_ciudad(ciudad, fecha, hora):
    url = f"https://api.geoapify.com/v1/geocode/search?text={ciudad}&apiKey={API_KEY}"
    try:
        response = requests.get(url, timeout=10)  # Timeout para evitar demoras
        if response.status_code == 200:
            datos = response.json()
            if datos.get("features"):
                opciones = [{
                    "nombre": resultado["properties"]["formatted"],
                    "lat": resultado["properties"]["lat"],
                    "lon": resultado["properties"]["lon"],
                    "pais": resultado["properties"].get("country", "")
                }
                for resultado in datos["features"]]
                return opciones
            return {"error": "Ciudad no encontrada"}
        return {"error": f"Error en la consulta: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Timeout en la consulta"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}

def obtener_zona_horaria(coordenadas, fecha):
    """
    Obtiene la zona horaria usando el archivo time_zone.csv y ajusta para horario de verano/invierno
    basado en las coordenadas y la fecha, considerando hemisferio norte/sur
    """
    try:
        lat = coordenadas["lat"]
        lon = coordenadas["lon"]
        fecha_obj = datetime.strptime(fecha, "%Y-%m-%d")
        
        # Determinar hemisferio (norte o sur)
        hemisferio = "norte" if lat >= 0 else "sur"
        
        # Verificar si la fecha está en horario de verano
        # Esta función necesita ser más precisa para fechas históricas
        is_dst = determinar_horario_verano(fecha_obj, hemisferio, coordenadas)
        
        # Buscar en el CSV por aproximación de longitud
        estimated_offset = round(lon / 15)
        
        # Ajustar para países específicos con información conocida
        pais = coordenadas.get("pais", "").lower()
        abbr = "UTC"
        tz_name = "UTC"
        offset = estimated_offset  # valor por defecto
        
        if "spain" in pais or "españa" in pais:
            tz_name = "Europe/Madrid"
            abbr = "CET"
            abbr_dst = "CEST"
            offset = 1
            if is_dst:
                offset = 2
                abbr = abbr_dst
        elif "argentina" in pais:
            tz_name = "America/Argentina/Buenos_Aires"
            abbr = "ART"
            offset = -3
            # Argentina no usa DST actualmente
        elif "mexico" in pais or "méxico" in pais:
            tz_name = "America/Mexico_City"
            abbr = "CST"
            abbr_dst = "CDT"
            offset = -6
            if is_dst:
                offset = -5
                abbr = abbr_dst
        else:
            # Buscar en el CSV la zona más cercana a la longitud estimada
            closest_zone = None
            min_diff = float('inf')
            
            if time_zone_df:
                for zone in time_zone_df:
                    # Los offsets en el CSV están en segundos, convertir a horas
                    csv_offset = zone['utc_offset'] / 3600
                    diff = abs(csv_offset - estimated_offset)
                    
                    if diff < min_diff:
                        min_diff = diff
                        closest_zone = zone
                
                if closest_zone:
                    offset = closest_zone['utc_offset'] / 3600
                    abbr = closest_zone['abbreviation']
                    tz_name = closest_zone['timezone']
                    
                    # Ajustar por DST si corresponde
                    if is_dst and closest_zone['dst'] == 1:
                        offset += 1
            else:
                # Si no hay datos en el CSV, usar la estimación por longitud
                offset = estimated_offset
                abbr = f"GMT{offset:+d}"
                tz_name = f"Estimated/GMT{offset:+d}"
        
        print(f"Zona horaria determinada: {tz_name}, offset: {offset}, DST: {is_dst}")
        
        return {
            "name": tz_name,
            "offset": offset,
            "abbreviation_STD": abbr,
            "abbreviation_DST": abbr,
            "is_dst": is_dst,
            "hemisphere": hemisferio
        }
    
    except Exception as e:
        print(f"Error obteniendo zona horaria: {str(e)}")
        # Si hay un error, devolver un mensaje claro
        print("Error en obtención de zona horaria, usando estimación basada en longitud")
        
        try:
            # Estimar zona horaria basada en longitud
            lon = coordenadas["lon"]
            estimated_offset = round(lon / 15)  # 15 grados = 1 hora
            
            # Para ciudades conocidas, usar valores predeterminados
            pais = coordenadas.get("pais", "").lower()
            
            if "spain" in pais or "españa" in pais:
                estimated_offset = 1
            elif "argentina" in pais:
                estimated_offset = -3
            elif "mexico" in pais or "méxico" in pais:
                estimated_offset = -6
            elif "united states" in pais or "estados unidos" in pais:
                # Aproximación basada en longitud para EEUU
                if lon < -100:
                    estimated_offset = -8  # Pacífico
                elif lon < -90:
                    estimated_offset = -7  # Montaña
                elif lon < -75:
                    estimated_offset = -6  # Central
                else:
                    estimated_offset = -5  # Este
            
            return {
                "name": f"GMT{estimated_offset:+d}",
                "offset": estimated_offset,
                "abbreviation_STD": f"GMT{estimated_offset:+d}",
                "abbreviation_DST": f"GMT{estimated_offset:+d}",
                "is_dst": False,
                "hemisphere": "norte" if coordenadas["lat"] >= 0 else "sur",
                "lon": lon  # Añadir longitud para referencia
            }
        except Exception as inner_e:
            print(f"Error en estimación de zona horaria: {str(inner_e)}")
            # Valor por defecto UTC si todo falla
            return {
                "name": "UTC",
                "offset": 0,
                "abbreviation_STD": "UTC",
                "abbreviation_DST": "UTC",
                "is_dst": False,
                "hemisphere": "norte",
                "estimated": True
            }

def determinar_horario_verano(fecha, hemisferio, coordenadas):
    """
    Determina si una fecha está en horario de verano (DST)
    Basado en reglas históricas y específicas por país
    """
    año = fecha.year
    mes = fecha.month
    dia = fecha.day
    
    # Obtener país
    pais = coordenadas.get("pais", "").lower()
    
    # Reglas específicas para España
    if "spain" in pais or "españa" in pais:
        # España antes de 1974: no había DST
        if año < 1974:
            return False
        elif año >= 1974 and año <= 1975:
            # En 1974-1975, DST fue del 13 de abril al 6 de octubre
            if (mes > 4 and mes < 10) or (mes == 4 and dia >= 13) or (mes == 10 and dia <= 6):
                return True
            return False
        elif año >= 1976 and año <= 1996:
            # Reglas más genéricas para 1976-1996
            # Primavera a otoño, aproximadamente marzo/abril a septiembre/octubre
            if mes > 3 and mes < 10:
                return True
            return False
        else:
            # Desde 1997: Regla actual de la UE - último domingo de marzo a último domingo de octubre
            if mes > 3 and mes < 10:
                return True
            # Marzo: último domingo
            elif mes == 3 and dia >= 25:  # Aproximación al último domingo
                return True
            # Octubre: último domingo
            elif mes == 10 and dia <= 25:  # Aproximación al último domingo
                return True
            return False
    
    # Reglas para otros países
    # Hemisferio Norte (Europa, América del Norte, Asia)
    elif hemisferio == "norte":
        # La mayoría de los países del hemisferio norte siguen este patrón
        # Horario de verano: finales de marzo a finales de octubre
        if año < 1970:
            # Antes de 1970 era menos común el DST
            return False
        
        if mes > 3 and mes < 10:
            return True
        elif mes == 3 and dia >= 25:  # Aproximación al último domingo de marzo
            return True
        elif mes == 10 and dia <= 25:  # Aproximación al último domingo de octubre
            return True
        return False
    
    # Hemisferio Sur (Australia, Nueva Zelanda, Sudamérica)
    else:
        # Muchos países del hemisferio sur no utilizan DST
        # Algunos que sí lo utilizan: Australia, Nueva Zelanda, Chile, Paraguay
        
        # Lista de países conocidos del hemisferio sur con DST
        south_dst_countries = ["australia", "new zealand", "nueva zelanda", "chile", "paraguay"]
        
        # Si no está en la lista, asumimos que no usa DST
        pais_usa_dst = any(country in pais for country in south_dst_countries)
        if not pais_usa_dst:
            return False
            
        # Horario de verano: finales de octubre a finales de marzo
        if mes < 3 or mes > 10:
            return True
        elif mes == 3 and dia <= 25:  # Aproximación al último domingo de marzo
            return True
        elif mes == 10 and dia >= 25:  # Aproximación al último domingo de octubre
            return True
        return False

def convertir_a_utc(fecha, hora, timezone_info):
    """
    Convierte fecha y hora local a UTC considerando zona horaria y DST
    Para cálculos astrológicos correctos, debemos asegurarnos de que la hora UTC sea precisa
    """
    try:
        # Combinar fecha y hora en un objeto datetime
        fecha_hora_str = f"{fecha} {hora}"
        dt_local = datetime.strptime(fecha_hora_str, "%Y-%m-%d %H:%M")
        
        # Obtener offset en horas desde la API de zona horaria
        # Si estamos en DST, la API ya incluye ese offset
        offset_hours = timezone_info["offset"]
        
        print(f"Offset de zona horaria: {offset_hours} horas")
        print(f"Hora local ingresada: {dt_local}")
        
        # Crear un timezone con el offset
        tz = timezone(timedelta(hours=offset_hours))
        
        # Aplicar timezone al datetime
        dt_local_with_tz = dt_local.replace(tzinfo=tz)
        
        # Convertir a UTC
        dt_utc = dt_local_with_tz.astimezone(timezone.utc)
        
        print(f"Hora convertida a UTC: {dt_utc}")
        return dt_utc
    except Exception as e:
        print(f"Error en conversión a UTC: {str(e)}")
        # Si falla, usar la hora proporcionada con offset manual aproximado
        dt_local = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M")
        
        # Intentar estimar offset basado en longitud si no tenemos zona horaria
        if "lon" in timezone_info:
            lon = timezone_info["lon"]
            est_offset = round(lon / 15)  # 15 grados = 1 hora
            est_tz = timezone(timedelta(hours=est_offset))
            dt_with_tz = dt_local.replace(tzinfo=est_tz)
            return dt_with_tz.astimezone(timezone.utc)
        
        # Fallback: asumir UTC
        return dt_local.replace(tzinfo=timezone.utc)

def calculate_positions_with_utc(utc_datetime, lat=None, lon=None):
    """
    Calcula posiciones planetarias con un datetime UTC
    Asegura que el tiempo se ajusta correctamente según la zona horaria
    """
    try:
        # Usar el datetime UTC directamente
        print(f"Calculando posiciones para UTC: {utc_datetime}")
        t = ts.from_datetime(utc_datetime)
        earth = eph['earth']
        
        positions = []
        bodies = {
            'SOL': eph['sun'],
            'LUNA': eph['moon'],
            'MERCURIO': eph['mercury'],
            'VENUS': eph['venus'],
            'MARTE': eph['mars'],
            'JÚPITER': eph['jupiter barycenter'],
            'SATURNO': eph['saturn barycenter'],
            'URANO': eph['uranus barycenter'],
            'NEPTUNO': eph['neptune barycenter'],
            'PLUTÓN': eph['pluto barycenter']
        }
        
        for body_name, body in bodies.items():
            pos = earth.at(t).observe(body).apparent()
            lat_ecl, lon_ecl, dist = pos.ecliptic_latlon(epoch='date')
            
            longitude = float(lon_ecl.degrees) % 360
            sign = get_sign(longitude)
            
            # Calcular grados y minutos dentro del signo
            degrees, minutes = calculate_sign_degrees(longitude, sign)
            sign_position = f"{degrees}° {minutes:02d}'"
            
            positions.append({
                "name": body_name,
                "longitude": longitude,
                "sign": sign,
                "sign_position": sign_position  # Nueva propiedad con la posición en el signo
            })
        
        if lat is not None and lon is not None:
            asc, mc = calculate_asc_mc(t, lat, lon)
            
            asc_sign = get_sign(asc)
            asc_degrees, asc_minutes = calculate_sign_degrees(asc, asc_sign)
            asc_sign_position = f"{asc_degrees}° {asc_minutes:02d}'"
            
            mc_sign = get_sign(mc)
            mc_degrees, mc_minutes = calculate_sign_degrees(mc, mc_sign)
            mc_sign_position = f"{mc_degrees}° {mc_minutes:02d}'"
            
            positions.append({
                "name": "ASC",
                "longitude": float(asc),
                "sign": asc_sign,
                "sign_position": asc_sign_position
            })
            
            positions.append({
                "name": "MC",
                "longitude": float(mc),
                "sign": mc_sign,
                "sign_position": mc_sign_position
            })
        
        return positions
    except Exception as e:
        print(f"Error calculando posiciones: {str(e)}")
        # No lanzar excepción, simplemente retornar un error formateado
        print("Usando método alternativo de cálculo")
        try:
            return calculate_positions(
                utc_datetime.strftime("%d/%m/%Y"),
                utc_datetime.strftime("%H:%M"),
                lat,
                lon
            )
        except Exception as inner_e:
            print(f"Error en método alternativo: {str(inner_e)}")
            return []

def calculate_positions(date_str, time_str, lat=None, lon=None):
    try:
        if '-' in date_str:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            date_str = date_obj.strftime("%d/%m/%Y")
            
        local_dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
        spain_tz = timezone(timedelta(hours=1))
        local_dt = local_dt.replace(tzinfo=spain_tz)
        utc_dt = local_dt.astimezone(timezone.utc)
        
        t = ts.from_datetime(utc_dt)
        earth = eph['earth']
        
        positions = []
        bodies = {
            'SOL': eph['sun'],
            'LUNA': eph['moon'],
            'MERCURIO': eph['mercury'],
            'VENUS': eph['venus'],
            'MARTE': eph['mars'],
            'JÚPITER': eph['jupiter barycenter'],
            'SATURNO': eph['saturn barycenter'],
            'URANO': eph['uranus barycenter'],
            'NEPTUNO': eph['neptune barycenter'],
            'PLUTÓN': eph['pluto barycenter']
        }
        
        for body_name, body in bodies.items():
            pos = earth.at(t).observe(body).apparent()
            lat_ecl, lon_ecl, dist = pos.ecliptic_latlon(epoch='date')
            
            longitude = float(lon_ecl.degrees) % 360
            sign = get_sign(longitude)
            
            # Calcular grados y minutos dentro del signo
            degrees, minutes = calculate_sign_degrees(longitude, sign)
            sign_position = f"{degrees}° {minutes:02d}'"
            
            positions.append({
                "name": body_name,
                "longitude": longitude,
                "sign": sign,
                "sign_position": sign_position  # Nueva propiedad con la posición en el signo
            })
        
        if lat is not None and lon is not None:
            asc, mc = calculate_asc_mc(t, lat, lon)
            
            asc_sign = get_sign(asc)
            asc_degrees, asc_minutes = calculate_sign_degrees(asc, asc_sign)
            asc_sign_position = f"{asc_degrees}° {asc_minutes:02d}'"
            
            mc_sign = get_sign(mc)
            mc_degrees, mc_minutes = calculate_sign_degrees(mc, mc_sign)
            mc_sign_position = f"{mc_degrees}° {mc_minutes:02d}'"
            
            positions.append({
                "name": "ASC",
                "longitude": float(asc),
                "sign": asc_sign,
                "sign_position": asc_sign_position
            })
            
            positions.append({
                "name": "MC",
                "longitude": float(mc),
                "sign": mc_sign,
                "sign_position": mc_sign_position
            })
        
        return positions
    except Exception as e:
        print(f"Error calculando posiciones: {str(e)}")
        return []

# Reemplazar la función calculate_asc_mc existente con esta nueva versión
def calculate_asc_mc(t, lat, lon):
    """
    Calcula Ascendente y Medio Cielo usando Swiss Ephemeris si está disponible,
    sino usa la aproximación con Skyfield
    """
    try:
        if SWISSEPH_AVAILABLE:
            # Usar Swiss Ephemeris para cálculos precisos
            julian_day = t.tt
            houses_data = swe.houses(julian_day, lat, lon, b'A')  # Sistema Placidus
            
            # houses_data[1] contiene ASC, MC, ARMC, Vertex
            asc_mc_data = houses_data[1]
            ascendente = asc_mc_data[0]  # Ascendente
            mc = asc_mc_data[1]          # Medio Cielo
            
            print(f"Swiss Ephemeris - ASC: {ascendente:.4f}°, MC: {mc:.4f}°")
            return ascendente, mc
        else:
            # Usar el método original con Skyfield como fallback
            return calculate_asc_mc_skyfield(t, lat, lon)
            
    except Exception as e:
        print(f"Error en calculate_asc_mc con Swiss Ephemeris: {str(e)}")
        # Si falla, usar el método de Skyfield
        return calculate_asc_mc_skyfield(t, lat, lon)

def calculate_asc_mc_skyfield(t, lat, lon):
    """Método original con Skyfield como fallback"""
    try:
        # Obtener tiempo sideral en Greenwich en grados
        gst = t.gast
        # Tiempo sideral local = GST + longitud
        lst = (gst * 15 + lon) % 360
        # MC es directamente el LST
        mc = lst % 360
        
        # Calcular Ascendente
        lat_rad = np.radians(lat)
        ra_rad = np.radians(lst)
        eps_rad = np.radians(23.4367)  # Oblicuidad de la eclíptica (aprox.)
        
        # Fórmula para el Ascendente
        tan_asc = np.cos(ra_rad) / (np.sin(ra_rad) * np.cos(eps_rad) + np.tan(lat_rad) * np.sin(eps_rad))
        asc = np.degrees(np.arctan(-tan_asc))
        
        # Ajustar cuadrante
        if 0 <= lst <= 180:
            if np.cos(ra_rad) > 0:
                asc = (asc + 180) % 360
        else:
            if np.cos(ra_rad) < 0:
                asc = (asc + 180) % 360
                
        asc = asc % 360
        
        print(f"Skyfield fallback - ASC: {asc:.4f}°, MC: {mc:.4f}°")
        return asc, mc
        
    except Exception as e:
        print(f"Error en calculate_asc_mc_skyfield: {str(e)}")
        # Valores por defecto en caso de error
        return 0, 0

def get_sign(longitude):
    longitude = float(longitude) % 360
    signs = [
        ("ARIES", 354.00, 36.00),
        ("TAURO", 30.00, 30.00),
        ("GÉMINIS", 60.00, 30.00),
        ("CÁNCER", 90.00, 30.00),
        ("LEO", 120.00, 30.00),
        ("VIRGO", 150.00, 36.00),
        ("LIBRA", 186.00, 24.00),
        ("ESCORPIO", 210.00, 30.00),
        ("OFIUCO", 240.00, 12.00),
        ("SAGITARIO", 252.00, 18.00),
        ("CAPRICORNIO", 270.00, 36.00),
        ("ACUARIO", 306.00, 18.00),
        ("PEGASO", 324.00, 6.00),
        ("PISCIS", 330.00, 24.00)
    ]
    
    for name, start, length in signs:
        end = (start + length) % 360
        if start <= longitude < end:
            return name
        elif start > 330 and (longitude >= start or longitude < end):
            # Caso especial para signos que cruzan 0° (como Aries)
            return name
    
    return "ARIES"  # Valor por defecto

# Nueva función para calcular grados dentro del signo
def calculate_sign_degrees(longitude, sign_name):
    longitude = float(longitude) % 360
    signs = [
        ("ARIES", 354.00, 36.00),
        ("TAURO", 30.00, 30.00),
        ("GÉMINIS", 60.00, 30.00),
        ("CÁNCER", 90.00, 30.00),
        ("LEO", 120.00, 30.00),
        ("VIRGO", 150.00, 36.00),
        ("LIBRA", 186.00, 24.00),
        ("ESCORPIO", 210.00, 30.00),
        ("OFIUCO", 240.00, 12.00),
        ("SAGITARIO", 252.00, 18.00),
        ("CAPRICORNIO", 270.00, 36.00),
        ("ACUARIO", 306.00, 18.00),
        ("PEGASO", 324.00, 6.00),
        ("PISCIS", 330.00, 24.00)
    ]
    
    # Encontrar el signo en la lista
    sign_info = next((s for s in signs if s[0] == sign_name), None)
    if not sign_info:
        return 0, 0  # Si no se encuentra el signo, devolver 0, 0
    
    start = sign_info[1]
    length = sign_info[2]
    
    # Calcular grados dentro del signo
    if start > 330 and longitude < start:  # Caso especial para signos que cruzan 0°
        degrees_in_sign = longitude + (360 - start)
    else:
        degrees_in_sign = longitude - start
    
    # Asegurarse de que esté dentro del rango del signo
    if degrees_in_sign < 0:
        degrees_in_sign += 360
    if degrees_in_sign >= length:
        degrees_in_sign = degrees_in_sign % length
    
    # Convertir a grados y minutos
    degrees = int(degrees_in_sign)
    minutes = int((degrees_in_sign - degrees) * 60)
    
    return degrees, minutes

def calculate_positions_aspects(positions):
    aspects = []
    traditional_planets = ["SOL", "LUNA", "MERCURIO", "VENUS", "MARTE", "JÚPITER", "SATURNO"]
    
    def calculate_angle(pos1, pos2):
        diff = abs(pos1 - pos2) % 360
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def determine_aspect_type(angle):
        orb = 2
        
        if (abs(angle) <= orb or 
            abs(angle - 60) <= orb or 
            abs(angle - 120) <= orb or
            abs(angle - 180) <= orb):
            return "Armónico Relevante"
        elif (abs(angle - 30) <= orb or
              abs(angle - 90) <= orb or
              abs(angle - 150) <= orb):
            return "Inarmónico Relevante"
        elif any(abs(angle - a) <= orb for a in [12, 24, 36, 48, 72, 84, 96, 108, 132, 144, 156, 168]):
            return "Armónico"
        elif any(abs(angle - a) <= orb for a in [6, 18, 42, 54, 66, 78, 102, 114, 126, 138, 162, 174]):
            return "Inarmónico"
            
        return None

    asc_position = next((p for p in positions if p["name"] == "ASC"), None)
    
    for i, pos1 in enumerate(positions):
        if pos1["name"] not in traditional_planets:
            continue
            
        for pos2 in positions[i+1:]:
            if pos2["name"] not in traditional_planets:
                continue
                
            angle = calculate_angle(pos1["longitude"], pos2["longitude"])
            aspect_type = determine_aspect_type(angle)
            
            if aspect_type:
                aspect = f"{pos1['name']} {aspect_type} {pos2['name']} ({angle:.2f}°)"
                aspects.append(aspect)
        
        # Aspectos con el ASC
        if asc_position:
            angle = calculate_angle(pos1["longitude"], asc_position["longitude"])
            aspect_type = determine_aspect_type(angle)
            
            if aspect_type:
                aspect = f"{pos1['name']} {aspect_type} ASC ({angle:.2f}°)"
                aspects.append(aspect)
    
    return aspects

def is_dry_birth(positions):
    """
    Determina si un nacimiento es seco basado en la posición del Sol.
    Es seco cuando el Sol está entre las casas 6 y 11 (inclusive).
    """
    sun_pos = next((p for p in positions if p["name"] == "SOL"), None)
    asc_pos = next((p for p in positions if p["name"] == "ASC"), None)
    
    if not sun_pos or not asc_pos:
        return False  # Valor predeterminado
    
    # Calcular la casa del Sol relativa al Ascendente
    diff = (sun_pos["longitude"] - asc_pos["longitude"]) % 360
    house = (diff // 30) + 1
    
    # Es seco si el Sol está en las casas 6 a 11
    return 6 <= house <= 11

def calculate_dignity(planet_name, longitude):
    total_points = 0
    sign = get_sign(longitude)
    
    dignities = {
        'SOL': {
            'domicilio': ['ESCORPIO', 'GÉMINIS', 'PEGASO'], 
            'exaltacion': ['LEO', 'ARIES', 'CAPRICORNIO', 'VIRGO'], 
            'caida': ['CÁNCER', 'PISCIS', 'LIBRA', 'ACUARIO', 'OFIUCO'], 
            'exilio': ['TAURO', 'SAGITARIO']  # TAURO está en exilio para el SOL
        },
        'LUNA': {
            'domicilio': ['TAURO', 'SAGITARIO'], 
            'exaltacion': ['CÁNCER', 'PISCIS', 'LIBRA', 'ACUARIO', 'OFIUCO'], 
            'caida': ['LEO', 'ARIES', 'CAPRICORNIO', 'VIRGO'], 
            'exilio': ['ESCORPIO', 'GÉMINIS', 'PEGASO']
        },
        'MERCURIO': {
            'domicilio': ['LEO', 'ARIES', 'ESCORPIO', 'PEGASO'], 
            'exaltacion': ['GÉMINIS', 'CAPRICORNIO', 'VIRGO'], 
            'caida': ['TAURO', 'LIBRA', 'ACUARIO', 'OFIUCO'], 
            'exilio': ['CÁNCER', 'PISCIS', 'SAGITARIO']
        },
        'VENUS': {
            'domicilio': ['CÁNCER', 'PISCIS', 'SAGITARIO'], 
            'exaltacion': ['TAURO', 'LIBRA', 'ACUARIO', 'OFIUCO'], 
            'caida': ['GÉMINIS', 'CAPRICORNIO', 'VIRGO'], 
            'exilio': ['LEO', 'ARIES', 'ESCORPIO', 'PEGASO']
        },
        'MARTE': {
            'domicilio': ['GÉMINIS', 'CAPRICORNIO', 'VIRGO'], 
            'exaltacion': ['LEO', 'ARIES', 'ESCORPIO', 'PEGASO'], 
            'caida': ['CÁNCER', 'PISCIS', 'SAGITARIO'], 
            'exilio': ['TAURO', 'LIBRA', 'ACUARIO', 'OFIUCO']
        },
        'JÚPITER': {  # Con acento
            'domicilio': ['TAURO', 'LIBRA', 'ACUARIO', 'OFIUCO'], 
            'exaltacion': ['CÁNCER', 'PISCIS', 'SAGITARIO'], 
            'caida': ['LEO', 'ARIES', 'ESCORPIO', 'PEGASO'], 
            'exilio': ['GÉMINIS', 'CAPRICORNIO', 'VIRGO']
        },
        'SATURNO': {
            'domicilio': ['LEO', 'ARIES', 'LIBRA', 'ACUARIO'], 
            'exaltacion': ['OFIUCO', 'GÉMINIS', 'SAGITARIO'], 
            'caida': ['TAURO', 'ESCORPIO', 'PEGASO'], 
            'exilio': ['CÁNCER', 'PISCIS', 'CAPRICORNIO', 'VIRGO']
        },
    }

    if planet_name in dignities:
        if sign in dignities[planet_name]["exaltacion"]:
            total_points += 6
        if sign in dignities[planet_name]["domicilio"]:
            total_points += 3
        if sign in dignities[planet_name]["caida"]:
            total_points += 0
        if sign in dignities[planet_name]["exilio"]:
            total_points += 3
            
    return total_points

def is_angular(longitude):
    specific_degrees = [354.00, 30.00, 60.00, 90.00, 120.00, 150.00, 186.00, 210.00, 
                       240.00, 252.00, 270.00, 306.00, 324.00, 330.00]
    orb = 1.00
    degree_in_sign = float(longitude) % 360
    
    for degree in specific_degrees:
        if abs(degree_in_sign - degree) <= orb:
            return 6
            
    return 0

def get_house_number(longitude, asc_longitude):
    """Calcula la casa desde el Ascendente."""
    diff = (longitude - asc_longitude) % 360
    house = 1 + (int(diff / 30))
    if house > 12:
        house = house - 12
    return house

def calculate_planet_aspects(planet_name, aspects_list):
    """
    Suma los puntos de los aspectos ya calculados para el planeta.
    """
    total = 0
    for aspect in aspects_list:
        if planet_name in aspect:  # Si el aspecto es de este planeta
            if "Armónico Relevante" in aspect:
                total += 6
            elif "Inarmónico Relevante" in aspect:
                total += -6
            elif "Armónico" in aspect:
                total += 1
            elif "Inarmónico" in aspect:
                total += -1
    return total

def calculate_dignity_table(positions, aspects_list):
    table = []
    total_points = 0
    
    # Nueva estructura con las dignidades de las casas según la especificación
    houses_rulers = {
        "SOL": {
            "exaltacion": [1, 5, 6, 10],
            "domicilio": [3, 8],
            "exilio": [2, 9],
            "caida": [4, 7, 11, 12]
        },
        "LUNA": {
            "exaltacion": [4, 7, 11, 12],
            "domicilio": [2, 9],
            "exilio": [3, 8],
            "caida": [1, 5, 6, 10]
        },
        "MERCURIO": {
            "exaltacion": [3, 6, 10],
            "domicilio": [5, 1, 8],
            "exilio": [4, 9, 12],
            "caida": [2, 7, 11]
        },
        "VENUS": {
            "exaltacion": [2, 7, 11],
            "domicilio": [4, 9, 12],
            "exilio": [5, 1, 8],
            "caida": [3, 6, 10]
        },
        "MARTE": {
            "exaltacion": [5, 1, 8],
            "domicilio": [3, 6, 10],
            "exilio": [2, 7, 11],
            "caida": [4, 9, 12]
        },
        "JÚPITER": {
            "exaltacion": [4, 9, 12],
            "domicilio": [2, 7, 11],
            "exilio": [3, 6, 10],
            "caida": [5, 1, 8]
        },
        "SATURNO": {
            "exaltacion": [3, 9],
            "domicilio": [1, 5, 7, 11],
            "exilio": [4, 6, 10, 12],
            "caida": [2, 8]
        }
    }
    
    # Valores de puntos para cada tipo de dignidad
    dignity_points = {
        "exaltacion": 6,
        "domicilio": 3,
        "exilio": 3,
        "caida": 0
    }
    
    asc_pos = next((p for p in positions if p["name"] == "ASC"), None)
    
    if not asc_pos:
        return {"tabla": [], "total_general": 0}
    
    for position in positions:
        if position["name"] in houses_rulers:
            house_num = get_house_number(position["longitude"], asc_pos["longitude"])
            
            # Calcular los puntos de dignidad basados en la posición de la casa
            house_points = 0
            dignity_type = ""
            
            for dtype, houses in houses_rulers[position["name"]].items():
                if house_num in houses:
                    house_points = dignity_points[dtype]
                    dignity_type = dtype
                    break
            
            # Calcular otros puntos como antes
            dignity_points_sign = calculate_dignity(position["name"], position["longitude"])
            angular_points = is_angular(position["longitude"]) 
            aspect_points = calculate_planet_aspects(position["name"], aspects_list)
            
            # Calcular el total para este planeta
            planet_total = house_points + dignity_points_sign + angular_points + aspect_points
            total_points += planet_total
            
            table.append({
                "planeta": position["name"],
                "signo": position["sign"],
                "casa": house_num,
                "puntos_casa": house_points,
                "puntos_dignidad": dignity_points_sign,
                "puntos_angular": angular_points,
                "puntos_aspectos": aspect_points,
                "total_planeta": planet_total
            })
    
    return {
        "tabla": table,
        "total_general": total_points
    }

SIGNS_BY_ELEMENT = {
    "AIRE": ["GÉMINIS", "ACUARIO", "OFIUCO", "LIBRA"],
    "TIERRA": ["TAURO", "CAPRICORNIO", "VIRGO"],
    "AGUA": ["ESCORPIO", "CÁNCER", "PISCIS", "PEGASO"],
    "FUEGO": ["ARIES", "LEO", "SAGITARIO"]
}

ELEMENT_BY_SIGN = {sign: element for element, signs in SIGNS_BY_ELEMENT.items() for sign in signs}

TRIPLICITIES = {
    "AIRE": {
        "humedo": "MERCURIO",    # Regente de Géminis
        "seco": "SATURNO",       # Regente de Ofiuco
        "participativo": "JÚPITER"  # Regente de Pegaso
    },
    "TIERRA": {
        "humedo": "VENUS",        # Regente de Tauro
        "seco": "MERCURIO",         # Regente de Virgo
        "participativo": "LUNA"  # Regente de Libra
    },
    "FUEGO": {
        "humedo": "SOL",         # Regente de Leo
        "seco": "SATURNO",       # Regente de Ofiuco
        "participativo": "MARTE"    # Regente de Aries
    },
    "AGUA": {
        "humedo": "LUNA",     # Regente de Cáncer
        "seco": "MARTE",         # Regente de Escorpio
        "participativo": "JÚPITER"     # Regente de Piscis
    }
}

HOUSE_MEANINGS = [
    "ÓRGANO DE LA INTELIGENCIA",
    "OBJETO DEL SUSTENTO",
    "OBJETO DE LA INTELIGENCIA",
    "MENTE",
    "INTELIGENCIA",
    "OBJETO DE LA MENTE",
    "OBJETO DE LA RELACIÓN",
    "ÓRGANO DE LA MENTE",
    "ÓRGANO DE LA RELACIÓN",
    "UNE EL ÓRGANO DEL SUSTENTO CON EL OBJETO DE LA RELACIÓN",
    "UNE EL OBJETO DEL SUSTENTO CON EL ÓRGANO DE LA RELACIÓN",
    "ÓRGANO DEL SUSTENTO"
]

def get_element_for_sign(sign):
    """
    Determina el elemento de un signo zodiacal.
    """
    return ELEMENT_BY_SIGN.get(sign, "FUEGO")  # FUEGO como valor predeterminado

def get_triplicity_rulers_for_sign(sign, is_dry_birth):
    """
    Obtiene los regentes de triplicidad para un signo dado.
    """
    element = get_element_for_sign(sign)
    if not element:
        # Si no se encuentra elemento, usar FUEGO por defecto
        element = "FUEGO"
    
    rulers = TRIPLICITIES[element]
    
    return {
        "regente1": rulers["humedo"],
        "regente2": rulers["seco"],
        "regente3": rulers["participativo"]
    }

def calculate_houses_with_triplicities(positions, is_dry_birth):
    """
    Calcula la tabla de casas con sus triplicidades.
    """
    asc_pos = next((p for p in positions if p["name"] == "ASC"), None)
    
    if not asc_pos:
        return []  # Si no hay ascendente, no podemos calcular casas
    
    houses_table = []
    
    for i in range(12):
        # Usar el Ascendente como punto de partida para las casas
        house_cusp = (asc_pos["longitude"] + (i * 30.00)) % 360
        sign = get_sign(house_cusp)
        element = get_element_for_sign(sign)
        triplicity_rulers = get_triplicity_rulers_for_sign(sign, is_dry_birth)
        
        house_data = {
            "house_number": i + 1,
            "element": element,
            "sign": sign,
            "cusp_longitude": f"{house_cusp:.2f}°",
            "meaning": HOUSE_MEANINGS[i],
            "triplicity_rulers": triplicity_rulers
        }
        
        houses_table.append(house_data)
    
    return houses_table

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/cities', methods=['GET'])
def get_cities():
    ciudad = request.args.get("ciudad")
    if not ciudad:
        return jsonify({"error": "Debes proporcionar una ciudad"}), 400

    print(f"Búsqueda recibida para ciudad: {ciudad}")
    
    # API key de Geoapify
    api_key = API_KEY
    
    # Usar la API de Geoapify para autocompletado de ciudades
    url = f"https://api.geoapify.com/v1/geocode/autocomplete?text={ciudad}&apiKey={api_key}&limit=20"
    
    try:
        # Hacer la petición a la API
        response = requests.get(url, timeout=10)
        print(f"Estado de respuesta Geoapify: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error en la API: {response.text}")
            raise Exception(f"Error en la API: {response.status_code}")
            
        data = response.json()
        
        # Mostrar la respuesta completa para depuración
        print(f"Respuesta completa de API: {data}")
        
        # Crear lista de ciudades encontradas
        ciudades = []
        
        # Verificar si hay resultados
        if "features" in data and len(data["features"]) > 0:
            print(f"Número de resultados: {len(data['features'])}")
            
            for feature in data["features"]:
                props = feature["properties"]
                # Formatear el nombre de la ciudad con país
                nombre_ciudad = props.get("formatted", "")
                if nombre_ciudad:
                    print(f"Ciudad encontrada: {nombre_ciudad}")
                    ciudades.append(nombre_ciudad)
        else:
            print("No se encontraron resultados en la API")
        
        # Si no hay resultados, generar algunas opciones
        if not ciudades:
            print("Generando opciones")
            ciudades = [
                f"{ciudad}, España",
                f"{ciudad}, México",
                f"{ciudad}, Argentina",
                f"{ciudad}, Estados Unidos",
                f"{ciudad}, Colombia"
            ]
        
        print(f"Total ciudades a devolver: {len(ciudades)}")
        print(f"Ciudades encontradas: {ciudades}")
        
        return jsonify({"ciudades": ciudades})
        
    except Exception as e:
        print(f"Error en búsqueda de ciudades: {str(e)}")
        # En caso de error, generar algunas opciones
        ciudades = [
            f"{ciudad}, España",
            f"{ciudad}, México",
            f"{ciudad}, Argentina",
            f"{ciudad}, Estados Unidos",
            f"{ciudad}, Colombia"
        ]
        
        return jsonify({"ciudades": ciudades})

@app.route('/calculate', methods=['POST', 'OPTIONS'])
def calculate():
    if request.method == 'OPTIONS':
        # Responder a la solicitud preflight de CORS
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        data = request.get_json()
        if not data or not data.get('city'):
            return jsonify({"error": "Ciudad no especificada"}), 400
            
        city_data = obtener_datos_ciudad(data['city'], data['date'], data['time'])
        
        if isinstance(city_data, dict) and "error" in city_data:
            return jsonify(city_data), 400
            
        if isinstance(city_data, list) and len(city_data) > 0:
            city_data = city_data[0]
        else:
            return jsonify({"error": "No se pudo obtener información de la ciudad"}), 400
        
        try:
            # Obtener zona horaria para las coordenadas
            timezone_info = obtener_zona_horaria(city_data, data['date'])
            
            # Convertir fecha y hora local a UTC
            utc_datetime = convertir_a_utc(data['date'], data['time'], timezone_info)
            
            # Calcular posiciones con el datetime UTC
            positions = calculate_positions_with_utc(utc_datetime, city_data["lat"], city_data["lon"])
            
            # Calcular aspectos entre posiciones
            aspects = calculate_positions_aspects(positions)
            
            # Calcular tabla de dignidades
            dignity_table = calculate_dignity_table(positions, aspects)
            
            # Determinar si es nacimiento seco o húmedo
            birth_type = "seco" if is_dry_birth(positions) else "húmedo"
            
            # Calcular casas y triplicidades
            houses_table = calculate_houses_with_triplicities(positions, birth_type == "seco")

            # Construir respuesta
            response = {
                "positions": positions,
                "coordinates": {
                    "latitude": city_data["lat"],
                    "longitude": city_data["lon"]
                },
                "city": city_data["nombre"],
                "timezone": timezone_info,
                "local_time": f"{data['date']} {data['time']}",
                "utc_time": utc_datetime.strftime("%Y-%m-%d %H:%M"),
                "aspects": aspects,
                "dignity_table": dignity_table,
                "houses_analysis": {
                    "houses": houses_table,
                    "birth_type": birth_type
                }
            }
            
            return jsonify(response)
            
        except Exception as timezone_error:
            print(f"Error con zona horaria: {str(timezone_error)}")
            # Si hay error con la zona horaria, usar la función original
            positions = calculate_positions(
                data['date'],
                data['time'],
                city_data["lat"],
                city_data["lon"]
            )
            
            # Cálculos adicionales con el método original
            aspects = calculate_positions_aspects(positions)
            dignity_table = calculate_dignity_table(positions, aspects)
            birth_type = "seco" if is_dry_birth(positions) else "húmedo"
            houses_table = calculate_houses_with_triplicities(positions, birth_type == "seco")
            
            return jsonify({
                "positions": positions,
                "coordinates": {
                    "latitude": city_data["lat"],
                    "longitude": city_data["lon"]
                },
                "city": city_data["nombre"],
                "error_timezone": f"No se pudo determinar la zona horaria: {str(timezone_error)}",
                "aspects": aspects,
                "dignity_table": dignity_table,
                "houses_analysis": {
                    "houses": houses_table,
                    "birth_type": birth_type
                }
            })
        
    except Exception as e:
        print(f"Error general: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\nIniciando servidor de carta astral optimizado...")
    preload_resources()
    print("Servidor iniciando en modo producción")
    app.run(host='0.0.0.0', port=10005, debug=False)