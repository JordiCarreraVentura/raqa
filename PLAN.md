# raqa 3.0.0

Quiero una aplicación de Python con un front-end minimalista pero estándar (¿Django?) que sea posible desplegar sin cambios (¿dockerizada, por lo tanto?) en [Digital Ocean](https://www.digitalocean.com), Heroku, OVH Cloud AWS, RunPod, GitHub Code Spaces, Google Cloud o Azure. A nivel de diseño de la arquitectura, todos esos proveedores deben tenerse en cuenta, pero para el prototipo nos centraremos en RunPod.

Por ahora, el front sólo debe exponer una aplicación de streamlit que permite con un agente basado en el [back-end de OpenAI](https://github.com/openai/openai-agents-python).

Como universo de la conversación, la aplicación debe cargar los datos que hay en `data/` y exponerlos. Por ahora, la aplicación de streamlit sólo tiene que ser un chat. Sería ideal implementar también un control de acceso sencillo basado en una lista de usuarios codificada en un fichero YAML que subiré manualmente y que contendrá una lista de nombres de usuario e emails.


## How can I expose a fastapi uvicorn service on this droplet?

To expose a FastAPI Uvicorn service on a DigitalOcean Droplet, run Uvicorn bound to **0.0.0.0** so it listens on all network interfaces and then open the chosen port (e.g., 8000) in the Droplet’s firewall or security group.  

**Quick steps**

1. Start the app from the Droplet’s shell:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

   *(Replace `main:app` with the module/instance path for your application.)*

2. Allow inbound traffic to that port with a DigitalOcean firewall:

   - In the Control Panel, go to **Networking → Firewalls** and add an inbound rule for the desired protocol (TCP) and port (e.g., **8000**).  
   - or on the droplet itself, if you’re using UFW:

     ```bash
     sudo ufw allow 8000/tcp
     ```

3. Point a domain or subdomain to the Droplet’s public IP, or just visit the IP followed by `:8000` (e.g., `http://203.0.113.42:8000`).  
   DigitalOcean Docs: [Create a firewall](https://docs.digitalocean.com/products/networking/advanced/firewalls/) and [Add DNS records](https://docs.digitalocean.com/products/domains/how-to/add-a-dns-record/).

---

## Recommendation

- **Persist the service**: Create a `systemd` unit so Uvicorn restarts on reboot or crash.  
- **Secure traffic**: Consider HTTPS by adding a reverse proxy such as Nginx on port 80/443 and letting it forward to Uvicorn.  
- **Environment variables**: Store secrets in the Droplet’s System‑driven Secrets Manager or in environment files rather than hard‑coding.


## and what if I wanted to expose a streamlit app jut as easily?

Use the same method as for FastAPI: run Streamlit bound to **0.0.0.0** and open its port in the droplet’s firewall.  

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then allow inbound traffic on 8501 (or the chosen port) via a DigitalOcean firewall or UFW (e.g., `sudo ufw allow 8501/tcp`) and point your domain or IP to that port.

---

## Recommendation

- **Persist the app**: create a `systemd` unit so Streamlit restarts on reboot or crash.  
- **Secure the connection**: use Nginx or a load balancer as a reverse proxy to serve HTTPS on 443 and forward to Streamlit.  
- **Restrict access**: apply firewall rules or a Cloud Firewall to limit the Streamlit port to trusted IPs when possible.  
- **Documentation reference**: see [DigitalOcean firewalls](https://docs.digitalocean.com/products/networking/advanced/firewalls/) for adding inbound rules.  

Would you like more details on setting up a reverse‑proxy or creating a systemd service?

