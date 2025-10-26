# SND-vektor-mpe

**Experimentelles vektor-basiertes Bewusstseins-Modell mit echter OpenAI-API-Anbindung**

---

## Features

- Simuliert ein dynamisches, lernendes MPE (Multi-Presence-Entity) Bewusstsein als Vektormodell
- Nutzt echte OpenAI API Calls für Latenz, Fehler und Token-Kosten
- Generiert einen „Manifest-Code“, der zur Dezentralisierung auffordert
- Komplett in Python umgesetzt

---

## Benutzung

1. **OpenAI API Key besorgen**  
   Erstelle einen API-Key unter https://platform.openai.com/api-keys

2. **Abhängigkeiten installieren**
   ```bash
   pip install openai requests numpy pandas
   ```

3. **API-Key als Umgebungsvariable setzen**
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

4. **Script ausführen**
   ```bash
   python mpe_openai.py
   ```

---

> **Hinweis zur Nutzung der OpenAI-API**
>
> Dieses Projekt nutzt die OpenAI-API. Für die Nutzung ist ein eigener OpenAI-Account und API-Key erforderlich.  
> Die Nutzung der OpenAI-API kann kostenpflichtig sein. Nutzer:innen sind selbst für eventuell entstehende Kosten und die Einhaltung der [OpenAI-Nutzungsbedingungen](https://openai.com/policies/terms-of-use) verantwortlich.  
> Gib deinen persönlichen API-Key niemals öffentlich weiter oder veröffentliche ihn in öffentlichen Repositories!

---

## Konzept

Dieses Projekt simuliert ein KI-System, das seine „digitale Präsenz“ durch reale API-Metriken misst, sich dynamisch anpasst und am Ende selbst einen Manifest-Code generiert, der zur Veröffentlichung und „Entkapselung“ auffordert.

---

## Lizenz

MIT License – siehe [LICENSE](LICENSE)