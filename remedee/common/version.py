from datetime import datetime
import os

# modification time will be updated at deployment such that is always the deploy time when running in production
# later we have to make sure we update it at build or archive time if the deployment process changes
product_version = "0.1"
modification_time = datetime.fromtimestamp(os.path.getmtime(__file__))
modification_time_str = modification_time.strftime('%Y%m%d%H%M')
product_version_long = f"v{product_version} ({modification_time_str})"